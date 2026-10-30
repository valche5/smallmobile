//! Le réseau sur un GPU NVIDIA par CUDA (`--device cuda`), à la place de Vulkan : même anneau de
//! positions et même file d'ordre que `vulkan::Engine` (le fil « gpu » de `gpu.rs` ne voit pas la
//! différence), mais en mémoire de l'hôte épinglée et lue directement par le GPU.
//!
//! Un calcul (`launch`) range les positions de la file par réseau, leurs régions à la suite (pas de
//! remplissage à 48), puis enchaîne sur un flot CUDA : de petits noyaux (`KERNELS`, compilés au
//! démarrage par NVRTC) pour rassembler les entrées, la moyenne des voisines, les moyennes et
//! maximums par position, biais et ReLU ; cuBLAS pour les produits. Les trois grands produits de
//! chaque couche en bf16 (h, la moyenne des voisines et relu(n1) arrondis au plus proche, sommes en
//! f32), comme le calcul bf16 du processeur ; les autres en f32. La fin de chaque calcul est un
//! événement que le fil « gpu » surveille (`finished`).
//!
//! Les bibliothèques (pilote, cuBLAS, NVRTC) sont chargées à l'exécution : le programme se compile
//! sans CUDA, et `--device cuda` dit ce qui manque sur la machine.

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::path::Path;
use std::sync::Mutex;

use cudarc::cublas::{result as blas, sys as bsys};
use cudarc::driver::{result as drv, sys};

use crate::encode::{GLOBAL, NODE, REGIONS};
use crate::moves::{GLOBAL_KEYS, KEYS};
use crate::net::{Position, REGION_KINDS as KINDS};
use crate::vulkan::{MAX_BATCH, ORDER, RING, STEP_SLOTS};

const OUT: usize = KEYS + 10;
/// Lignes de régions au plus dans un calcul.
const ROWS: usize = MAX_BATCH * REGIONS;
/// Réseaux au plus sur un moteur.
const NETS: usize = 8;
/// Le tampon épinglé d'un calcul, en entiers : les places de ses positions, puis leurs décalages
/// (ceux de chaque réseau ont une case de plus).
const STAGE: usize = 2 * MAX_BATCH + NETS;

/// Les étapes d'un calcul, chronométrées avec `LAB14_CUDA_PHASES=1` (`Prof`).
pub const PHASES: [&str; 14] = [
    "entrées (gather)",
    "node_in",
    "glob_in",
    "voisines",
    "n1, partie",
    "n1, régions",
    "mid",
    "n2",
    "résiduel des régions",
    "pool",
    "g1",
    "g2 et résiduel",
    "têtes",
    "sorties (finish)",
];
/// Un calcul chronométré sur `PROF_EVERY` : ses événements sont attendus aussitôt (le fil « gpu »
/// s'arrête le temps du calcul), pour savoir où passe le temps, pas pour la vitesse.
const PROF_EVERY: usize = 8;
/// Événements d'un calcul chronométré : un au départ, un après chaque étape (~100 pour 7 couches).
const PROF_EVENTS: usize = 256;

/// Le chronométrage des étapes (`LAB14_CUDA_PHASES=1`) : les événements, l'étape de chacun, et les
/// totaux, affichés à la fin du programme (`Drop`).
struct Prof {
    events: Vec<sys::CUevent>,
    marks: Vec<usize>,
    ms: [f64; PHASES.len()],
    calcs: usize,
    positions: usize,
    rows: usize,
}

impl Prof {
    /// La fin de l'étape `phase`, sur le flot `stream`.
    unsafe fn mark(&mut self, stream: sys::CUstream, phase: usize) -> Result<(), String> {
        let at = self.marks.len() + 1;
        assert!(at < PROF_EVENTS, "trop d'étapes chronométrées");
        self.marks.push(phase);
        unsafe { drv::event::record(self.events[at], stream).map_err(err("événement")) }
    }
}

/// L'anneau des entrées, octets : comme `vulkan::InputLayout`.
#[derive(Clone, Copy)]
struct Layout {
    node: usize,
    glob: usize,
    adj: usize,
    regions: usize,
    nets: usize,
    order: usize,
    size: usize,
}

impl Layout {
    fn new() -> Layout {
        let up = |x: usize| x.next_multiple_of(16);
        let node = 0;
        let glob = up(node + RING * REGIONS * NODE);
        let adj = up(glob + RING * GLOBAL);
        let regions = up(adj + RING * REGIONS * 8);
        let nets = up(regions + RING);
        let order = up(nets + RING);
        Layout { node, glob, adj, regions, nets, order, size: order + ORDER * 4 }
    }
}

const KERNELS: &str = r#"
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned long long u64;

// bf16 au plus proche (pair en cas d'égalité), comme `net.rs`
__device__ u16 bf16(float x) {
    unsigned u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (u16)(u >> 16);
}

// une position par bloc : ses régions à la suite (`off`), entrées mises à l'échelle
extern "C" __global__ void gather(int count, const int* cells, const int* off, const u8* input,
        long long node_at, long long glob_at, long long adj_at, long long regions_at,
        const float* node_scale, const float* glob_scale,
        float* xnode, float* xglob, u64* adj, int* row_pos) {
    int p = blockIdx.x;
    if (p >= count) return;
    long long cell = cells[p];
    int r = input[regions_at + cell];
    int base = off[p];
    const u8* node = input + node_at + cell * 48 * 93;
    for (int i = threadIdx.x; i < r * 93; i += blockDim.x)
        xnode[(long long)base * 93 + i] = (float)node[i] * node_scale[i % 93];
    const u8* glob = input + glob_at + cell * 820;
    for (int i = threadIdx.x; i < 820; i += blockDim.x)
        xglob[(long long)p * 820 + i] = (float)glob[i] * glob_scale[i];
    const u64* bits = (const u64*)(input + adj_at + cell * 48 * 8);
    for (int i = threadIdx.x; i < r; i += blockDim.x) {
        adj[base + i] = bits[i];
        row_pos[base + i] = p;
    }
}

// y[i, j] += b[j] ; relu si demandé
extern "C" __global__ void bias(int rows, int n, float* y, const float* b, int relu) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * n) return;
    float v = y[i] + b[i % n];
    y[i] = relu ? fmaxf(v, 0.0f) : v;
}

// [h | moyenne des voisines] en bf16, une ligne de région par bloc, h colonnes par fil
extern "C" __global__ void neighbours(int rows, int h, const float* x, const u64* adj, const int* row_pos,
        const int* off, u16* cat) {
    int i = blockIdx.x;
    if (i >= rows) return;
    int base = off[row_pos[i]];
    u64 bits = adj[i];
    int deg = __popcll(bits);
    float inv = 1.0f / (float)(deg > 0 ? deg : 1);
    for (int c = threadIdx.x; c < h; c += blockDim.x) {
        float s = 0.0f;
        u64 b = bits;
        while (b) {
            int j = __ffsll(b) - 1;
            b &= b - 1;
            s += x[(long long)(base + j) * h + c];
        }
        cat[(long long)i * 2 * h + c] = bf16(x[(long long)i * h + c]);
        cat[(long long)i * 2 * h + h + c] = bf16(s * inv);
    }
}

// relu(n1) en bf16 : produit des régions plus la part de la partie (biais compris)
extern "C" __global__ void mid(int rows, int h, const float* y, const float* gpart, const int* row_pos, u16* out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * h) return;
    long long r = i / h;
    float v = y[i] + gpart[(long long)row_pos[r] * h + i % h];
    out[i] = bf16(fmaxf(v, 0.0f));
}

// x += y + b
extern "C" __global__ void residual(int rows, int n, float* x, const float* y, const float* b) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * n) return;
    x[i] += y[i] + b[i % n];
}

// par position : [g | moyenne des régions | maximum] (with_max), ou [g | moyenne]
extern "C" __global__ void pool(int count, int h, const float* x, const int* off, const float* g, float* out,
        int with_max) {
    int p = blockIdx.x;
    if (p >= count) return;
    int a = off[p], b = off[p + 1];
    int w = with_max ? 3 : 2;
    for (int c = threadIdx.x; c < h; c += blockDim.x) {
        float s = 0.0f, m = -3.402823466e38f;
        for (int i = a; i < b; i++) {
            float v = x[(long long)i * h + c];
            s += v;
            m = fmaxf(m, v);
        }
        out[(long long)p * w * h + c] = g[(long long)p * h + c];
        out[(long long)p * w * h + h + c] = s / (float)(b - a);
        if (with_max) out[(long long)p * w * h + 2 * h + c] = m;
    }
}

// les sorties de chaque position dans son anneau : logits des clés (moins l'infini pour les régions
// absentes), puis les 10 valeurs
extern "C" __global__ void finish(int count, const int* cells, const int* off, const float* gl, const float* gl_b,
        const float* gpol, const float* np_b, const float* npol, const float* v, const float* v_b, float* output) {
    int p = blockIdx.x;
    if (p >= count) return;
    float* out = output + (long long)cells[p] * OUTN;
    int a = off[p], r = off[p + 1] - a;
    for (int k = threadIdx.x; k < OUTN; k += blockDim.x) {
        float y;
        if (k < 32) {
            y = gl[p * 32 + k] + gl_b[k];
        } else if (k < KEYSN) {
            int kind = (k - 32) / 64, i = (k - 32) % 64;
            y = i < r ? npol[(long long)(a + i) * 9 + kind] + gpol[p * 9 + kind] + np_b[kind] : -__int_as_float(0x7f800000);
        } else {
            y = v[p * 10 + k - KEYSN] + v_b[k - KEYSN];
        }
        out[k] = y;
    }
}
"#;

/// Une couche linéaire sur le GPU : poids (sorties × entrées, comme le fichier) en f32, et en bf16
/// pour les grands produits.
struct Lin {
    out: usize,
    inp: usize,
    w: sys::CUdeviceptr,
    w16: Option<sys::CUdeviceptr>,
    b: sys::CUdeviceptr,
}

struct NetW {
    node_in: Lin,
    glob_in: Lin,
    layers: Vec<[Lin; 4]>,
    glob_pol1: Lin,
    glob_pol2: Lin,
    node_pol: Lin,
    val1: Lin,
    val2: Lin,
    node_scale: sys::CUdeviceptr,
    glob_scale: sys::CUdeviceptr,
}

/// Les tampons d'un calcul, pour `ROWS` lignes de régions et `MAX_BATCH` positions.
struct Work {
    xnode: sys::CUdeviceptr,
    xglob: sys::CUdeviceptr,
    adj: sys::CUdeviceptr,
    row_pos: sys::CUdeviceptr,
    h: sys::CUdeviceptr,
    cat: sys::CUdeviceptr,
    y: sys::CUdeviceptr,
    midb: sys::CUdeviceptr,
    g: sys::CUdeviceptr,
    gpart: sys::CUdeviceptr,
    gcat: sys::CUdeviceptr,
    gmid: sys::CUdeviceptr,
    gtmp: sys::CUdeviceptr,
    gl: sys::CUdeviceptr,
    gpol: sys::CUdeviceptr,
    npol: sys::CUdeviceptr,
    v: sys::CUdeviceptr,
    /// la copie, dans la mémoire du GPU, des places et décalages du calcul (`launch`) : lus à chaque
    /// région par les noyaux, ils traversaient le PCIe depuis le tampon épinglé (`staging`), plus de la
    /// moitié du temps d'un calcul (voisines, 26 octobre)
    stage: sys::CUdeviceptr,
}

struct Kernels {
    gather: sys::CUfunction,
    bias: sys::CUfunction,
    neighbours: sys::CUfunction,
    mid: sys::CUfunction,
    residual: sys::CUfunction,
    pool: sys::CUfunction,
    finish: sys::CUfunction,
}

/// Le réseau sur un GPU NVIDIA, avec un anneau de `RING` positions (entrées et sorties).
pub struct Engine {
    ctx: sys::CUcontext,
    stream: sys::CUstream,
    blas: bsys::cublasHandle_t,
    kernels: Kernels,
    nets: Vec<NetW>,
    hidden: usize,
    work: Work,
    layout: Layout,
    /// anneau des entrées et file d'ordre, puis sorties (`RING · OUT` nombres) ; épinglés, lus et
    /// écrits par le GPU sur place
    input: *mut u8,
    input_dev: sys::CUdeviceptr,
    output: *mut f32,
    output_dev: sys::CUdeviceptr,
    /// par étape (`STEP_SLOTS`) : places et décalages des positions de son calcul (épinglés),
    /// l'événement de sa fin
    staging: Vec<(*mut i32, sys::CUdeviceptr)>,
    events: Vec<sys::CUevent>,
    name: String,
    /// un seul fil lance à la fois (le fil « gpu »)
    launching: Mutex<()>,
    /// `LAB14_CUDA_PHASES=1` : le chronométrage des étapes, et le compte des calculs
    prof: Option<Mutex<Prof>>,
    tick: std::sync::atomic::AtomicUsize,
}

// SAFETY : les pointeurs de l'hôte sont des tampons épinglés dont chaque position n'est écrite que
// par qui la tient (`gpu::Gpu`) ; les appels CUDA passent par le contexte, rendu courant à chaque fois
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

fn err<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> String {
    move |e| format!("CUDA, {what} : {e:?}")
}

/// Les tenseurs d'un fichier de poids : nom → (dimensions, nombres).
type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Lit un fichier SMNN1.
fn tensors(path: &Path) -> Result<Tensors, String> {
    // le processeur doit pouvoir le lire aussi : fichier complet et cohérent
    crate::net::Net::load(path).map_err(|e| format!("{} : {e}", path.display()))?;
    let bytes = std::fs::read(path).map_err(|e| format!("{} : {e}", path.display()))?;
    let mut f = &bytes[5..];
    let mut take = |n: usize| {
        let (head, rest) = f.split_at(n);
        f = rest;
        head
    };
    let word = |b: &[u8]| u32::from_le_bytes(b.try_into().expect("4 octets")) as usize;
    let count = word(take(4));
    let mut out = HashMap::new();
    for _ in 0..count {
        let len = word(take(4));
        let name = String::from_utf8_lossy(take(len)).into_owned();
        let ndim = word(take(4));
        let dims: Vec<usize> = (0..ndim).map(|_| word(take(4))).collect();
        let size: usize = dims.iter().product();
        let data = take(size * 4).as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
        out.insert(name, (dims, data));
    }
    Ok(out)
}

fn bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

unsafe fn upload<T>(data: &[T]) -> Result<sys::CUdeviceptr, String> {
    unsafe {
        let p = drv::malloc_sync(std::mem::size_of_val(data).max(4)).map_err(err("mémoire"))?;
        drv::memcpy_htod_sync(p, data).map_err(err("copie"))?;
        Ok(p)
    }
}

unsafe fn zeros(bytes: usize) -> Result<sys::CUdeviceptr, String> {
    unsafe {
        let p = drv::malloc_sync(bytes.max(4)).map_err(err("mémoire"))?;
        drv::memset_d8_sync(p, 0, bytes.max(4)).map_err(err("mémoire"))?;
        Ok(p)
    }
}

/// Mémoire de l'hôte épinglée, lue et écrite par le GPU sur place : son adresse côté hôte et GPU.
unsafe fn mapped(bytes: usize) -> Result<(*mut c_void, sys::CUdeviceptr), String> {
    unsafe {
        let host = drv::malloc_host(bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE)
            .map_err(err("mémoire épinglée"))?;
        std::ptr::write_bytes(host as *mut u8, 0, bytes);
        let mut dev = 0;
        sys::cuMemHostGetDevicePointer_v2(&mut dev, host, 0).result().map_err(err("mémoire épinglée"))?;
        Ok((host, dev))
    }
}

impl Engine {
    /// Ouvre le premier GPU CUDA avec les poids `nets` (de même forme) ; une position dit le sien à
    /// l'écriture (`write`).
    pub fn new_nets(nets: &[&Path]) -> Result<Engine, String> {
        // SAFETY : appels du pilote dans l'ordre de la documentation CUDA, le contexte rendu courant ;
        // chaque objet créé est détruit par `Drop`
        unsafe {
            drv::init().map_err(err("pilote (libcuda)"))?;
            let dev = drv::device::get(0).map_err(err("appareil"))?;
            let name = drv::device::get_name(dev).map_err(err("appareil"))?;
            let ctx = drv::primary_ctx::retain(dev).map_err(err("contexte"))?;
            drv::ctx::set_current(ctx).map_err(err("contexte"))?;
            let stream = drv::stream::create(drv::stream::StreamKind::NonBlocking).map_err(err("flot"))?;
            let blas = blas::create_handle().map_err(err("cuBLAS"))?;
            blas::set_stream(blas, stream as _).map_err(err("cuBLAS"))?;

            let (major, minor) = (
                drv::device::get_attribute(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                    .map_err(err("appareil"))?,
                drv::device::get_attribute(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                    .map_err(err("appareil"))?,
            );
            let arch: &'static str = Box::leak(format!("compute_{major}{minor}").into_boxed_str());
            let src = KERNELS.replace("OUTN", &OUT.to_string()).replace("KEYSN", &KEYS.to_string());
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(
                src,
                cudarc::nvrtc::CompileOptions { arch: Some(arch), ..Default::default() },
            )
            .map_err(err("NVRTC"))?;
            let image = CString::new(ptx.to_src()).map_err(err("PTX"))?;
            let module = drv::module::load_data(image.as_ptr() as *const c_void).map_err(err("module"))?;
            let function = |n: &str| -> Result<sys::CUfunction, String> {
                let n = CString::new(n).expect("nom");
                drv::module::get_function(module, n).map_err(err("noyau"))
            };
            let kernels = Kernels {
                gather: function("gather")?,
                bias: function("bias")?,
                neighbours: function("neighbours")?,
                mid: function("mid")?,
                residual: function("residual")?,
                pool: function("pool")?,
                finish: function("finish")?,
            };

            let mut loaded = Vec::new();
            let mut hidden = 0;
            for path in nets {
                let mut t = tensors(path)?;
                let node_scale = upload(&t.remove("node_scale").ok_or("node_scale manquant")?.1)?;
                let glob_scale = upload(&t.remove("global_scale").ok_or("global_scale manquant")?.1)?;
                let depth = (0..).take_while(|l| t.contains_key(&format!("layers.{l}.n1.weight"))).count();
                let mut lin = |name: &str, half: bool| -> Result<Lin, String> {
                    let missing = || format!("poids manquants : {name}");
                    let (dims, w) = t.remove(&format!("{name}.weight")).ok_or_else(missing)?;
                    let (_, b) = t.remove(&format!("{name}.bias")).ok_or_else(missing)?;
                    let w16 =
                        if half { Some(upload(&w.iter().map(|&x| bf16(x)).collect::<Vec<u16>>())?) } else { None };
                    Ok(Lin { out: dims[0], inp: dims[1], w: upload(&w)?, w16, b: upload(&b)? })
                };
                let node_in = lin("node_in", false)?;
                let glob_in = lin("glob_in", false)?;
                let mut layers = Vec::new();
                for l in 0..depth {
                    layers.push([
                        lin(&format!("layers.{l}.n1"), true)?,
                        lin(&format!("layers.{l}.n2"), true)?,
                        lin(&format!("layers.{l}.g1"), false)?,
                        lin(&format!("layers.{l}.g2"), false)?,
                    ]);
                }
                let w = NetW {
                    glob_pol1: lin("glob_pol1", false)?,
                    glob_pol2: lin("glob_pol2", false)?,
                    node_pol: lin("node_pol", false)?,
                    val1: lin("val1", false)?,
                    val2: lin("val2", false)?,
                    node_in,
                    glob_in,
                    layers,
                    node_scale,
                    glob_scale,
                };
                if hidden != 0 && w.node_in.out != hidden {
                    return Err("réseaux de largeurs différentes".into());
                }
                hidden = w.node_in.out;
                if w.node_in.inp != NODE
                    || w.glob_in.inp != GLOBAL
                    || w.glob_pol2.out != GLOBAL_KEYS
                    || w.node_pol.out != KINDS
                {
                    return Err("réseau d'une autre forme".into());
                }
                loaded.push(w);
            }
            let h = hidden;
            let (f, s) = (4, 2); // octets : f32, bf16
            let work = Work {
                xnode: zeros(ROWS * NODE * f)?,
                xglob: zeros(MAX_BATCH * GLOBAL * f)?,
                adj: zeros(ROWS * 8)?,
                row_pos: zeros(ROWS * 4)?,
                h: zeros(ROWS * h * f)?,
                cat: zeros(ROWS * 2 * h * s)?,
                y: zeros(ROWS * h * f)?,
                midb: zeros(ROWS * h * s)?,
                g: zeros(MAX_BATCH * h * f)?,
                gpart: zeros(MAX_BATCH * h * f)?,
                gcat: zeros(MAX_BATCH * 3 * h * f)?,
                gmid: zeros(MAX_BATCH * h * f)?,
                gtmp: zeros(MAX_BATCH * h * f)?,
                gl: zeros(MAX_BATCH * GLOBAL_KEYS * f)?,
                gpol: zeros(MAX_BATCH * KINDS * f)?,
                npol: zeros(ROWS * KINDS * f)?,
                v: zeros(MAX_BATCH * 10 * f)?,
                stage: zeros(STAGE * 4)?,
            };
            let layout = Layout::new();
            let (input, input_dev) = mapped(layout.size)?;
            let (output, output_dev) = mapped(RING * OUT * 4)?;
            if loaded.len() > NETS {
                return Err(format!("{NETS} réseaux au plus"));
            }
            let staging = (0..STEP_SLOTS)
                .map(|_| mapped(STAGE * 4).map(|(h, d)| (h as *mut i32, d)))
                .collect::<Result<Vec<_>, _>>()?;
            let events = (0..STEP_SLOTS)
                .map(|_| drv::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING).map_err(err("événement")))
                .collect::<Result<Vec<_>, _>>()?;
            let prof = match std::env::var_os("LAB14_CUDA_PHASES") {
                Some(_) => Some(Mutex::new(Prof {
                    events: (0..PROF_EVENTS)
                        .map(|_| drv::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).map_err(err("événement")))
                        .collect::<Result<Vec<_>, _>>()?,
                    marks: Vec::new(),
                    ms: [0.0; PHASES.len()],
                    calcs: 0,
                    positions: 0,
                    rows: 0,
                })),
                None => None,
            };
            Ok(Engine {
                ctx,
                stream,
                blas,
                kernels,
                nets: loaded,
                hidden,
                work,
                layout,
                input: input as *mut u8,
                input_dev,
                output: output as *mut f32,
                output_dev,
                staging,
                events,
                name,
                launching: Mutex::new(()),
                prof,
                tick: std::sync::atomic::AtomicUsize::new(0),
            })
        }
    }

    pub fn describe(&self) -> String {
        format!("GPU {} (CUDA, cuBLAS bf16)", self.name)
    }

    #[allow(clippy::mut_from_ref)]
    fn input(&self) -> &mut [u8] {
        // SAFETY : mémoire épinglée de `layout.size` octets ; chaque position n'est écrite que par qui
        // la tient (`gpu::Gpu`)
        unsafe { std::slice::from_raw_parts_mut(self.input, self.layout.size) }
    }

    /// Les sorties de la position `cell` de l'anneau : `OUT` nombres.
    pub(crate) fn output(&self, cell: usize) -> &[f32] {
        // SAFETY : mémoire épinglée de `RING · OUT` nombres
        unsafe { std::slice::from_raw_parts(self.output.add(cell * OUT), OUT) }
    }

    /// Écrit la position `p` à la place `cell` de l'anneau (comme `vulkan::Engine::write`).
    pub(crate) fn write<P: Position + ?Sized>(&self, cell: usize, p: &P, net: u8) {
        let (l, input) = (self.layout, self.input());
        let e = p.encoded();
        let node = l.node + cell * REGIONS * NODE;
        input[node..node + e.regions * NODE].copy_from_slice(e.node[..e.regions].as_flattened());
        let glob = l.glob + cell * GLOBAL;
        input[glob..glob + GLOBAL].copy_from_slice(&e.global);
        let adj = l.adj + cell * REGIONS * 8;
        for (i, out) in input[adj..adj + REGIONS * 8].as_chunks_mut::<8>().0.iter_mut().enumerate() {
            let bits = if i < e.regions { p.adj(i).iter().fold(0u64, |b, &j| b | 1 << j) } else { 0 };
            *out = bits.to_le_bytes();
        }
        input[l.regions + cell] = e.regions as u8;
        input[l.nets + cell] = net;
    }

    /// Range la place `cell` au numéro `at` de la file d'ordre.
    pub(crate) fn order(&self, at: usize, cell: usize) {
        let o = self.layout.order + at * 4;
        self.input()[o..o + 4].copy_from_slice(&(cell as u32).to_le_bytes());
    }

    unsafe fn run(
        &self,
        f: sys::CUfunction,
        grid: usize,
        block: usize,
        args: &mut [*mut c_void],
    ) -> Result<(), String> {
        if grid == 0 {
            return Ok(());
        }
        unsafe {
            drv::launch_kernel(f, (grid as u32, 1, 1), (block as u32, 1, 1), 0, self.stream, args).map_err(err("noyau"))
        }
    }

    /// `y (rows × w.out, f32) = x (rows × k) · W[:, from..from + k]ᵀ` ; `x` en bf16 si `half`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
        &self,
        w: &Lin,
        from: usize,
        k: usize,
        x: sys::CUdeviceptr,
        ldx: usize,
        rows: usize,
        y: sys::CUdeviceptr,
        half: bool,
    ) -> Result<(), String> {
        if rows == 0 {
            return Ok(());
        }
        let (one, zero) = (1.0f32, 0.0f32);
        let (wp, ty, size) = if half {
            (w.w16.expect("poids bf16"), bsys::cudaDataType::CUDA_R_16BF, 2)
        } else {
            (w.w, bsys::cudaDataType::CUDA_R_32F, 4)
        };
        // en colonnes (cuBLAS) : yᵀ (out × rows) = W (out × inp, rangé en lignes = Wᵀ en colonnes)ᵀ · xᵀ
        unsafe {
            blas::gemm_ex(
                self.blas,
                bsys::cublasOperation_t::CUBLAS_OP_T,
                bsys::cublasOperation_t::CUBLAS_OP_N,
                w.out as i32,
                rows as i32,
                k as i32,
                &one as *const f32 as *const c_void,
                (wp + (from * size) as u64) as *const c_void,
                ty,
                w.inp as i32,
                x as *const c_void,
                ty,
                ldx as i32,
                &zero as *const f32 as *const c_void,
                y as *mut c_void,
                bsys::cudaDataType::CUDA_R_32F,
                w.out as i32,
                bsys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bsys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(err("cuBLAS"))
        }
    }

    unsafe fn bias(
        &self,
        rows: usize,
        n: usize,
        y: sys::CUdeviceptr,
        b: sys::CUdeviceptr,
        relu: bool,
    ) -> Result<(), String> {
        let (mut rows_, mut n_, mut y_, mut b_, mut relu_) = (rows as i32, n as i32, y, b, relu as i32);
        let mut args = [arg(&mut rows_), arg(&mut n_), arg(&mut y_), arg(&mut b_), arg(&mut relu_)];
        unsafe { self.run(self.kernels.bias, (rows * n).div_ceil(256), 256, &mut args) }
    }

    /// Le réseau `w` sur `count` positions : leurs places `cells` et les décalages de leurs régions
    /// `off` (épinglés, côté GPU), `rows` lignes en tout.
    unsafe fn forward(
        &self,
        w: &NetW,
        count: usize,
        cells: sys::CUdeviceptr,
        off: sys::CUdeviceptr,
        rows: usize,
        mut prof: Option<&mut Prof>,
    ) -> Result<(), String> {
        let (h, k, l) = (self.hidden, &self.kernels, self.layout);
        let wk = &self.work;
        let stream = self.stream;
        // la fin d'une étape, si ce calcul est chronométré
        let mut mark = |phase: usize| -> Result<(), String> {
            match prof.as_deref_mut() {
                // SAFETY : événement de `Prof`, sur le flot du calcul
                Some(p) => unsafe { p.mark(stream, phase) },
                None => Ok(()),
            }
        };
        unsafe {
            let (mut count_, mut rows_, mut h_) = (count as i32, rows as i32, h as i32);
            let (mut cells_, mut off_, mut input_) = (cells, off, self.input_dev);
            let (mut node_at, mut glob_at, mut adj_at, mut regions_at) =
                (l.node as i64, l.glob as i64, l.adj as i64, l.regions as i64);
            let (mut ns, mut gs) = (w.node_scale, w.glob_scale);
            let (mut xnode, mut xglob, mut adj, mut row_pos) = (wk.xnode, wk.xglob, wk.adj, wk.row_pos);
            let mut args = [
                arg(&mut count_),
                arg(&mut cells_),
                arg(&mut off_),
                arg(&mut input_),
                arg(&mut node_at),
                arg(&mut glob_at),
                arg(&mut adj_at),
                arg(&mut regions_at),
                arg(&mut ns),
                arg(&mut gs),
                arg(&mut xnode),
                arg(&mut xglob),
                arg(&mut adj),
                arg(&mut row_pos),
            ];
            self.run(k.gather, count, 128, &mut args)?;
            mark(0)?;
            self.gemm(&w.node_in, 0, NODE, wk.xnode, NODE, rows, wk.h, false)?;
            self.bias(rows, h, wk.h, w.node_in.b, false)?;
            mark(1)?;
            self.gemm(&w.glob_in, 0, GLOBAL, wk.xglob, GLOBAL, count, wk.g, false)?;
            self.bias(count, h, wk.g, w.glob_in.b, false)?;
            mark(2)?;
            for [n1, n2, g1, g2] in &w.layers {
                let (mut x, mut adj, mut row_pos, mut off2, mut cat) = (wk.h, wk.adj, wk.row_pos, off, wk.cat);
                let mut args = [
                    arg(&mut rows_),
                    arg(&mut h_),
                    arg(&mut x),
                    arg(&mut adj),
                    arg(&mut row_pos),
                    arg(&mut off2),
                    arg(&mut cat),
                ];
                self.run(k.neighbours, rows, h.min(128), &mut args)?;
                mark(3)?;
                // la part de la partie, biais compris
                self.gemm(n1, 2 * h, h, wk.g, h, count, wk.gpart, false)?;
                self.bias(count, h, wk.gpart, n1.b, false)?;
                mark(4)?;
                self.gemm(n1, 0, 2 * h, wk.cat, 2 * h, rows, wk.y, true)?;
                mark(5)?;
                let (mut y, mut gpart, mut row_pos, mut midb) = (wk.y, wk.gpart, wk.row_pos, wk.midb);
                let mut args =
                    [arg(&mut rows_), arg(&mut h_), arg(&mut y), arg(&mut gpart), arg(&mut row_pos), arg(&mut midb)];
                self.run(k.mid, (rows * h).div_ceil(256), 256, &mut args)?;
                mark(6)?;
                self.gemm(n2, 0, h, wk.midb, h, rows, wk.y, true)?;
                mark(7)?;
                let (mut x, mut y, mut b) = (wk.h, wk.y, n2.b);
                let mut args = [arg(&mut rows_), arg(&mut h_), arg(&mut x), arg(&mut y), arg(&mut b)];
                self.run(k.residual, (rows * h).div_ceil(256), 256, &mut args)?;
                mark(8)?;
                let (mut x, mut off2, mut g, mut gcat, mut with_max) = (wk.h, off, wk.g, wk.gcat, 1i32);
                let mut args = [
                    arg(&mut count_),
                    arg(&mut h_),
                    arg(&mut x),
                    arg(&mut off2),
                    arg(&mut g),
                    arg(&mut gcat),
                    arg(&mut with_max),
                ];
                self.run(k.pool, count, h.min(128), &mut args)?;
                mark(9)?;
                self.gemm(g1, 0, 3 * h, wk.gcat, 3 * h, count, wk.gmid, false)?;
                self.bias(count, h, wk.gmid, g1.b, true)?;
                mark(10)?;
                self.gemm(g2, 0, h, wk.gmid, h, count, wk.gtmp, false)?;
                let (mut g, mut y, mut b) = (wk.g, wk.gtmp, g2.b);
                let mut args = [arg(&mut count_), arg(&mut h_), arg(&mut g), arg(&mut y), arg(&mut b)];
                self.run(k.residual, (count * h).div_ceil(256), 256, &mut args)?;
                mark(11)?;
            }
            // politique : clés de la partie, part commune des régions, régions
            self.gemm(&w.glob_pol1, 0, h, wk.g, h, count, wk.gmid, false)?;
            self.bias(count, h, wk.gmid, w.glob_pol1.b, true)?;
            self.gemm(&w.glob_pol2, 0, h, wk.gmid, h, count, wk.gl, false)?;
            self.gemm(&w.node_pol, h, h, wk.g, h, count, wk.gpol, false)?;
            self.gemm(&w.node_pol, 0, h, wk.h, h, rows, wk.npol, false)?;
            // valeur : val1([g, moyenne h])
            let (mut x, mut off2, mut g, mut gcat, mut with_max) = (wk.h, off, wk.g, wk.gcat, 0i32);
            let mut args = [
                arg(&mut count_),
                arg(&mut h_),
                arg(&mut x),
                arg(&mut off2),
                arg(&mut g),
                arg(&mut gcat),
                arg(&mut with_max),
            ];
            self.run(k.pool, count, h.min(128), &mut args)?;
            self.gemm(&w.val1, 0, 2 * h, wk.gcat, 2 * h, count, wk.gmid, false)?;
            self.bias(count, h, wk.gmid, w.val1.b, true)?;
            self.gemm(&w.val2, 0, h, wk.gmid, h, count, wk.v, false)?;
            mark(12)?;
            let (mut cells_, mut off2, mut gl, mut glb, mut gpol, mut npb, mut npol, mut v, mut vb, mut output) =
                (cells, off, wk.gl, w.glob_pol2.b, wk.gpol, w.node_pol.b, wk.npol, wk.v, w.val2.b, self.output_dev);
            let mut args = [
                arg(&mut count_),
                arg(&mut cells_),
                arg(&mut off2),
                arg(&mut gl),
                arg(&mut glb),
                arg(&mut gpol),
                arg(&mut npb),
                arg(&mut npol),
                arg(&mut v),
                arg(&mut vb),
                arg(&mut output),
            ];
            self.run(k.finish, count, 256, &mut args)?;
            mark(13)
        }
    }

    /// Lance le calcul des `count` positions de la file d'ordre à partir de son numéro `first`, à
    /// l'étape `slot` : rangées par réseau, chacun son passage.
    pub(crate) fn launch(&self, slot: usize, first: usize, count: usize) -> Result<(), String> {
        let _held = self.launching.lock().expect("verrou");
        assert!(count <= MAX_BATCH && first < ORDER, "calcul trop grand");
        let (l, input) = (self.layout, self.input());
        let cell_at = |t: usize| {
            let o = l.order + ((first + t) % ORDER) * 4;
            u32::from_le_bytes(input[o..o + 4].try_into().expect("4 octets")) as usize
        };
        // SAFETY : tampon épinglé de cette étape, libre (le fil « gpu » ne relance une étape qu'après sa fin)
        let (stage, _) = self.staging[slot];
        let stage = unsafe { std::slice::from_raw_parts_mut(stage, STAGE) };
        let (cells, offs) = stage.split_at_mut(MAX_BATCH);
        unsafe { drv::ctx::set_current(self.ctx).map_err(err("contexte"))? };
        // un calcul sur `PROF_EVERY` chronométré, étape par étape (`LAB14_CUDA_PHASES`)
        let tick = self.tick.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut prof =
            self.prof.as_ref().filter(|_| tick.is_multiple_of(PROF_EVERY)).map(|p| p.lock().expect("verrou"));
        if let Some(p) = prof.as_deref_mut() {
            p.marks.clear();
            // SAFETY : événement de `Prof`, sur le flot des calculs
            unsafe { drv::event::record(p.events[0], self.stream).map_err(err("événement"))? };
        }
        // les places et décalages de chaque réseau : ses positions à la suite de celles des réseaux
        // d'avant ; ses décalages (une case de plus que de positions) à la suite des leurs
        let mut start = 0;
        let mut passes = Vec::new();
        for n in 0..self.nets.len() {
            let (cells, offs) = (&mut cells[start..], &mut offs[start + n..]);
            let mut c = 0;
            offs[0] = 0;
            for t in 0..count {
                let cell = cell_at(t);
                if input[l.nets + cell] as usize == n {
                    cells[c] = cell as i32;
                    offs[c + 1] = offs[c] + input[l.regions + cell] as i32;
                    c += 1;
                }
            }
            if c > 0 {
                passes.push((n, c, offs[c] as usize, start * 4, (MAX_BATCH + start + n) * 4));
            }
            start += c;
        }
        // copiés dans la mémoire du GPU, avant les noyaux qui les lisent (même flot) ; le tampon
        // épinglé de cette étape reste intact jusqu'à la fin du calcul
        // SAFETY : tampon épinglé de `STAGE` entiers, copie vers un tampon du GPU de même taille
        unsafe { drv::memcpy_htod_async(self.work.stage, &stage[..], self.stream).map_err(err("copie"))? };
        let dev = self.work.stage;
        for (n, c, rows, at_cells, at_offs) in passes {
            // SAFETY : places et décalages copiés juste avant sur le même flot
            unsafe {
                self.forward(&self.nets[n], c, dev + at_cells as u64, dev + at_offs as u64, rows, prof.as_deref_mut())?
            };
            if let Some(p) = prof.as_deref_mut() {
                p.rows += rows;
            }
        }
        if let Some(p) = prof.as_deref_mut() {
            // SAFETY : événements de `Prof`, posés sur le flot de ce calcul
            unsafe {
                let last = p.events[p.marks.len()];
                drv::event::synchronize(last).map_err(err("événement"))?;
                for (i, &phase) in p.marks.iter().enumerate() {
                    p.ms[phase] += drv::event::elapsed(p.events[i], p.events[i + 1]).map_err(err("événement"))? as f64;
                }
            }
            p.calcs += 1;
            p.positions += count;
        }
        unsafe { drv::event::record(self.events[slot], self.stream).map_err(err("événement")) }
    }

    /// Le calcul de l'étape `slot` est-il fini ?
    pub(crate) fn finished(&self, slot: usize) -> Result<bool, String> {
        // SAFETY : événement créé par `new_nets`
        match unsafe { drv::event::query(self.events[slot]) } {
            Ok(()) => Ok(true),
            Err(e) if e.0 == sys::cudaError_enum::CUDA_ERROR_NOT_READY => Ok(false),
            Err(e) => Err(format!("CUDA, calcul : {e:?}")),
        }
    }
}

fn arg<T>(x: &mut T) -> *mut c_void {
    x as *mut T as *mut c_void
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(p) = &self.prof {
            let p = p.lock().expect("verrou");
            let total: f64 = p.ms.iter().sum();
            let (calcs, positions) = (p.calcs.max(1) as f64, p.positions.max(1) as f64);
            eprintln!(
                "étapes des calculs CUDA : {} calculs chronométrés, {:.0} positions et {:.0} régions par calcul, {:.0} µs par calcul, {:.2} µs par position",
                p.calcs,
                positions / calcs,
                p.rows as f64 / calcs,
                1e3 * total / calcs,
                1e3 * total / positions
            );
            for (name, ms) in PHASES.iter().zip(p.ms) {
                eprintln!(
                    "    {name:<22} {:8.1} µs par calcul  {:5.1} %",
                    1e3 * ms / calcs,
                    100.0 * ms / total.max(1e-9)
                );
            }
        }
        // SAFETY : plus aucun calcul en vol (le fil « gpu » est arrêté) ; objets créés par `new_nets`
        unsafe {
            let _ = drv::ctx::set_current(self.ctx);
            let _ = drv::stream::synchronize(self.stream);
            for &e in &self.events {
                let _ = drv::event::destroy(e);
            }
            if let Some(p) = &self.prof {
                for &e in &p.lock().expect("verrou").events {
                    let _ = drv::event::destroy(e);
                }
            }
            let _ = blas::destroy_handle(self.blas);
            let _ = drv::stream::destroy(self.stream);
            for &(h, _) in &self.staging {
                let _ = drv::free_host(h as *mut c_void);
            }
            let _ = drv::free_host(self.input as *mut c_void);
            let _ = drv::free_host(self.output as *mut c_void);
        }
    }
}
