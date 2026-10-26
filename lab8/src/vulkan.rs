//! L'encodeur de lab8 sur GPU par Vulkan (joueurs `plans:…,gpu=1`), lot par lot.
//!
//! Le calcul de `net::Net::encode`, en kernels successifs sur des lots de positions
//! (`lab8/shaders`) : entrées des jetons (`embed`), produits par les poids (`gemm`, bf16 sur les
//! matrices coopératives, sommes en f32), positions et jetons absents (`fixup`), `layernorm`,
//! `attention`, `pool`, et les sorties pour le processeur (`pack`) : la sortie de l'encodeur pour les
//! régions, `dec_reg`, les clés et valeurs de l'attention croisée du décodeur en bf16, les têtes de
//! valeur en f32. Le décodeur reste sur le processeur (`Net::step`, 0,3 ms par pas).
//!
//! Les produits arrondissent leurs entrées en bf16 : de petits écarts avec le processeur
//! (`lab8 check-gpu`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use ash::vk;
use smallmobile_brain8::net::{Memory, read_tensors};
use smallmobile_brain8::tokens::{GAME_IN, MARKET_IN, PLAYER_IN, PLAYERS, REGION_IN, RELATIONS, TOKENS};

use crate::encode::{Encoded, GLOBAL, NODE, REGIONS};

/// Octets d'une position dans le tampon des entrées (`common.glsl`).
const POS_BYTES: usize = REGIONS * NODE + GLOBAL + REGIONS * REGIONS + 4;
const _: () = assert!(POS_BYTES.is_multiple_of(4) && TOKENS == 60);

const SPV: [&[u8]; 7] = [
    include_bytes!(concat!(env!("OUT_DIR"), "/embed.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/fixup.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/layernorm.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/pool.spv")),
    include_bytes!(concat!(env!("OUT_DIR"), "/pack.spv")),
];
const EMBED: usize = 0;
const GEMM: usize = 1;
const FIXUP: usize = 2;
const LAYERNORM: usize = 3;
const ATTENTION: usize = 4;
const POOL: usize = 5;
const PACK: usize = 6;

/// `VK_KHR_shader_bfloat16` (absent d'ash 0.38), comme lab7.
#[repr(C)]
struct Bf16Features {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    shader_bfloat16_type: vk::Bool32,
    shader_bfloat16_dot_product: vk::Bool32,
    shader_bfloat16_cooperative_matrix: vk::Bool32,
}
const BF16_FEATURES: vk::StructureType = vk::StructureType::from_raw(1_000_141_000);

fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn from_bf16(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn up16(x: usize) -> usize {
    x.div_ceil(16) * 16
}

fn vk_err(what: &str) -> impl Fn(vk::Result) -> String + '_ {
    move |e| format!("Vulkan, {what} : {e}")
}

/// Durée cumulée (ns) et nombre d'appels de chaque sorte de kernel (`LAB8_GPU_PROFILE`).
type KernelTimes = [(f64, u64); 7];

/// Des positions à lire : encodage et distances de la carte.
type Items = Vec<(Encoded, Vec<u8>)>;

/// Une couche linéaire pour `gemm` : ses tuiles (dans `wb`), son biais (dans `w`), ses entrées et
/// sorties complétées à 16.
#[derive(Clone, Copy, Default)]
struct Lin {
    w: u32,
    b: u32,
    k: u32,
    n: u32,
}

#[derive(Clone, Copy, Default)]
struct Norm {
    g: u32,
    b: u32,
}

struct Layer {
    ln1: Norm,
    qkv: Lin,
    o: Lin,
    rel: u32,
    ln2: Norm,
    a: Lin,
    b: Lin,
}

/// Les poids préparés : f32 (`w`) et tuiles bf16 (`wb`), et où sont les parties.
struct Weights {
    w: Vec<f32>,
    wb: Vec<u16>,
    d: usize,
    heads: usize,
    dec: usize,
    inputs: [Lin; 4],
    scales: [u32; 4],
    player_index: u32,
    player_pos: u32,
    market_pos: u32,
    layers: Vec<Layer>,
    enc_ln: Norm,
    val: (Lin, Lin),
    next: (Lin, Lin),
    kv: Lin,
    reg: Lin,
}

struct Reader(HashMap<String, (Vec<usize>, Vec<f32>)>);

impl Reader {
    fn take(&mut self, name: &str) -> Result<Vec<f32>, String> {
        self.0.remove(name).map(|t| t.1).ok_or(format!("tenseur absent : {name}"))
    }

    fn has(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
}

impl Weights {
    fn load(path: &Path) -> Result<Weights, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{} : {e}", path.display()))?;
        let mut r = Reader(read_tensors(&bytes).map_err(|e| format!("{} : {e}", path.display()))?);
        let d = r.0.get("region_in.bias").map(|t| t.1.len()).ok_or("region_in absent")?;
        let mut w = Weights {
            w: Vec::new(),
            wb: Vec::new(),
            d,
            heads: 0,
            dec: 0,
            inputs: [Lin::default(); 4],
            scales: [0; 4],
            player_index: 0,
            player_pos: 0,
            market_pos: 0,
            layers: Vec::new(),
            enc_ln: Norm::default(),
            val: Default::default(),
            next: Default::default(),
            kv: Lin::default(),
            reg: Lin::default(),
        };
        for (i, (name, k)) in
            [("region_in", REGION_IN), ("player_in", PLAYER_IN), ("market_in", MARKET_IN), ("game_in", GAME_IN)]
                .into_iter()
                .enumerate()
        {
            w.inputs[i] = w.lin(&mut r, &[name], k)?;
        }
        for (i, name) in ["region_scale", "player_scale", "market_scale", "game_scale"].iter().enumerate() {
            let v = r.take(name)?;
            w.scales[i] = w.push(&v);
        }
        r.take("dyn_scale")?;
        let places: Vec<f32> = (0..PLAYERS).flat_map(player_places).collect();
        w.player_index = w.push(&places);
        let v = r.take("player_pos.weight")?;
        w.player_pos = w.push(&v);
        let v = r.take("market_pos.weight")?;
        w.market_pos = w.push(&v);
        let mut i = 0;
        while r.has(&format!("layers.{i}.ln1.weight")) {
            let p = format!("layers.{i}");
            let ln1 = w.norm(&mut r, &format!("{p}.ln1"))?;
            let qkv = w.lin(&mut r, &[&format!("{p}.attn.q"), &format!("{p}.attn.k"), &format!("{p}.attn.v")], d)?;
            let o = w.lin(&mut r, &[&format!("{p}.attn.o")], d)?;
            let rel_v = r.take(&format!("{p}.rel.weight"))?;
            w.heads = rel_v.len() / RELATIONS;
            let rel = w.push(&rel_v);
            let ln2 = w.norm(&mut r, &format!("{p}.ln2"))?;
            let a = w.lin(&mut r, &[&format!("{p}.mlp.a")], d)?;
            let b = w.lin(&mut r, &[&format!("{p}.mlp.b")], 4 * d)?;
            w.layers.push(Layer { ln1, qkv, o, rel, ln2, a, b });
            i += 1;
        }
        w.enc_ln = w.norm(&mut r, "enc_ln")?;
        w.val = (w.lin(&mut r, &["val1"], 2 * d)?, w.lin(&mut r, &["val2"], d)?);
        w.next = (w.lin(&mut r, &["next1"], 2 * d)?, w.lin(&mut r, &["next2"], d)?);
        let mut names = Vec::new();
        while r.has(&format!("dec_layers.{}.cross.k.weight", w.dec)) {
            names.push(format!("dec_layers.{}.cross.k", w.dec));
            names.push(format!("dec_layers.{}.cross.v", w.dec));
            w.dec += 1;
        }
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        w.kv = w.lin(&mut r, &names, d)?;
        w.reg = w.lin(&mut r, &["dec_reg"], d)?;
        if w.heads == 0 || d != 32 * w.heads {
            return Err(format!("moteur GPU : largeur {d} pour {} têtes (32 par tête attendus)", w.heads));
        }
        Ok(w)
    }

    fn push(&mut self, v: &[f32]) -> u32 {
        let at = self.w.len() as u32;
        self.w.extend_from_slice(v);
        at
    }

    fn norm(&mut self, r: &mut Reader, name: &str) -> Result<Norm, String> {
        let g = r.take(&format!("{name}.weight"))?;
        let b = r.take(&format!("{name}.bias"))?;
        Ok(Norm { g: self.push(&g), b: self.push(&b) })
    }

    /// Les couches `names` mises bout à bout (leurs sorties à la suite), de `k` entrées, en tuiles
    /// 16 × 16 par colonnes (`gemm.comp`), complétées de zéros.
    fn lin(&mut self, r: &mut Reader, names: &[&str], k: usize) -> Result<Lin, String> {
        let (mut weight, mut bias) = (Vec::new(), Vec::new());
        for name in names {
            weight.extend(r.take(&format!("{name}.weight"))?);
            bias.extend(r.take(&format!("{name}.bias"))?);
        }
        let n = bias.len();
        if weight.len() != n * k {
            return Err(format!("{names:?} : forme inattendue"));
        }
        let (n16, k16) = (up16(n), up16(k));
        let at = self.wb.len() as u32;
        for nt in 0..n16 / 16 {
            for kt in 0..k16 / 16 {
                for nn in 0..16 {
                    for kk in 0..16 {
                        let (o, i) = (nt * 16 + nn, kt * 16 + kk);
                        self.wb.push(if o < n && i < k { bf16(weight[o * k + i]) } else { 0 });
                    }
                }
            }
        }
        bias.resize(n16, 0.0);
        let b = self.push(&bias);
        Ok(Lin { w: at, b, k: k16 as u32, n: n16 as u32 })
    }
}

/// Les places, dans les entrées de la partie, des entrées du joueur relatif `r` (`tokens::player`).
fn player_places(r: usize) -> Vec<f32> {
    const G_PLAYER: usize = 67;
    const P: usize = 81;
    const G_HABITS: usize = 700;
    const HABITS: usize = 14;
    const G_ATTACKS: usize = 770;
    const G_RECENT: usize = 795;
    let mut v: Vec<usize> = (0..P).map(|i| G_PLAYER + r * P + i).collect();
    v.extend((0..HABITS).map(|i| G_HABITS + r * HABITS + i));
    v.extend((0..PLAYERS).map(|t| G_ATTACKS + r * PLAYERS + t));
    v.extend((0..PLAYERS).map(|t| G_ATTACKS + t * PLAYERS + r));
    v.extend((0..PLAYERS).map(|t| G_RECENT + r * PLAYERS + t));
    v.extend((0..PLAYERS).map(|t| G_RECENT + t * PLAYERS + r));
    assert_eq!(v.len(), PLAYER_IN);
    v.into_iter().map(|x| x as f32).collect()
}

/// Où sont les matrices des activations (f32, dans `act`), pour `max` positions.
struct Acts {
    region: u32,
    player: u32,
    market: u32,
    game: u32,
    x: u32,
    n: u32,
    qkv: u32,
    att: u32,
    h: u32,
    o: u32,
    pool: u32,
    v1: u32,
    v2: u32,
    n1: u32,
    n2: u32,
    kv: u32,
    reg: u32,
    size: usize,
}

impl Acts {
    fn new(max: usize, d: usize, dec: usize) -> Acts {
        let mut at = 0usize;
        let mut take = |n: usize| {
            let a = at as u32;
            at += n;
            a
        };
        let rows = max * TOKENS;
        Acts {
            region: take(max * REGIONS * 80),
            player: take(max * PLAYERS * 128),
            market: take(max * 6 * 48),
            game: take(max * 80),
            x: take(rows * d),
            n: take(rows * d),
            qkv: take(rows * 3 * d),
            att: take(rows * d),
            h: take(rows * 4 * d),
            o: take(rows * d),
            pool: take(max * 2 * d),
            v1: take(max * d),
            v2: take(max * 16),
            n1: take(max * d),
            n2: take(max * 16),
            kv: take(rows * 2 * dec * d),
            reg: take(max * REGIONS * d),
            size: at,
        }
    }
}

pub struct Encoder {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    modules: Vec<vk::ShaderModule>,
    buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    input: *mut u8,
    out16: *const u16,
    outf: *const f32,
    weights: Weights,
    acts: Acts,
    /// Nombres bf16 rendus par position (`pack.comp`).
    per16: usize,
    pub max: usize,
    name: String,
    /// Un lot à la fois.
    lock: Mutex<()>,
    /// `LAB8_GPU_PROFILE` : un horodatage après chaque kernel, et la durée cumulée de chaque sorte
    /// (nanosecondes, nombre), affichée à la fin.
    profile: Option<(vk::QueryPool, f64, Mutex<KernelTimes>)>,
    marks: Mutex<Vec<usize>>,
}

// SAFETY : les pointeurs sont ceux de la mémoire projetée, qui vit autant que l'`Encoder` ; un seul
// lot à la fois les touche (`lock`).
unsafe impl Send for Encoder {}
unsafe impl Sync for Encoder {}

impl Encoder {
    /// Le premier GPU dédié (ou celui dont le nom contient `LAB8_VULKAN_DEVICE`) et les poids `net`,
    /// pour des lots de `max` positions au plus.
    pub fn new(net: &Path, max: usize) -> Result<Encoder, String> {
        let weights = Weights::load(net)?;
        // SAFETY : appels Vulkan dans l'ordre de la spécification ; chaque objet créé est détruit
        // par `Drop`
        unsafe {
            // RADV 26.2, gfx1201 : les paires d'instructions VOPD lisent parfois un registre trop tôt
            // (lab7, journal du 23 octobre) : sans elles
            if std::env::var("ACO_DEBUG").map_or(true, |f| !f.contains("nosched-vopd")) {
                let flags = match std::env::var("ACO_DEBUG") {
                    Ok(f) if !f.is_empty() => format!("{f},nosched-vopd"),
                    _ => "nosched-vopd".to_string(),
                };
                // SAFETY : avant le pilote ; les autres fils ne lisent pas l'environnement
                std::env::set_var("ACO_DEBUG", flags);
            }
            let entry = ash::Entry::load().map_err(|e| format!("Vulkan introuvable : {e}"))?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
                .map_err(vk_err("instance"))?;
            let wanted = std::env::var("LAB8_VULKAN_DEVICE").ok();
            let mut best: Option<(u32, vk::PhysicalDevice, u32, String)> = None;
            for pd in instance.enumerate_physical_devices().map_err(vk_err("appareils"))? {
                let props = instance.get_physical_device_properties(pd);
                let name = props.device_name_as_c_str().unwrap_or(c"?").to_string_lossy().into_owned();
                let rank = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    _ => 2,
                };
                if wanted.as_ref().is_some_and(|w| !name.contains(w.as_str())) {
                    continue;
                }
                let families = instance.get_physical_device_queue_family_properties(pd);
                let family = families.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE));
                if let Some(family) = family
                    && best.as_ref().is_none_or(|(r, ..)| rank < *r)
                {
                    best = Some((rank, pd, family as u32, name));
                }
            }
            let Some((_, pd, family, name)) = best else {
                instance.destroy_instance(None);
                return Err("aucun appareil Vulkan qui calcule".into());
            };
            let priorities = [1.0];
            let queues =
                [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default().shader_float16(true).vulkan_memory_model(true);
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default().subgroup_size_control(true);
            let mut coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
            let mut b16 = Bf16Features {
                s_type: BF16_FEATURES,
                p_next: std::ptr::null_mut(),
                shader_bfloat16_type: vk::TRUE,
                shader_bfloat16_dot_product: vk::FALSE,
                shader_bfloat16_cooperative_matrix: vk::TRUE,
            };
            coop.p_next = &mut b16 as *mut Bf16Features as *mut std::ffi::c_void;
            v13.p_next = &mut coop as *mut _ as *mut std::ffi::c_void;
            v12.p_next = &mut v13 as *mut _ as *mut std::ffi::c_void;
            let extensions = [c"VK_KHR_cooperative_matrix".as_ptr(), c"VK_KHR_shader_bfloat16".as_ptr()];
            let mut info =
                vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&extensions);
            info.p_next = &v12 as *const _ as *const std::ffi::c_void;
            let device =
                instance.create_device(pd, &info, None).map_err(vk_err("appareil (bf16, matrices coopératives)"))?;
            let queue = device.get_device_queue(family, 0);
            let memory = instance.get_physical_device_memory_properties(pd);
            let acts = Acts::new(max, weights.d, weights.dec);
            let per16 = 2 * REGIONS * weights.d + 2 * weights.dec * TOKENS * weights.d;
            let mut e = Encoder {
                _entry: entry,
                instance,
                device,
                queue,
                command_pool: vk::CommandPool::null(),
                command: vk::CommandBuffer::null(),
                fence: vk::Fence::null(),
                descriptor_pool: vk::DescriptorPool::null(),
                set_layout: vk::DescriptorSetLayout::null(),
                set: vk::DescriptorSet::null(),
                layout: vk::PipelineLayout::null(),
                pipelines: Vec::new(),
                modules: Vec::new(),
                buffers: Vec::new(),
                input: std::ptr::null_mut(),
                out16: std::ptr::null(),
                outf: std::ptr::null(),
                weights,
                acts,
                per16,
                max,
                name,
                lock: Mutex::new(()),
                profile: None,
                marks: Mutex::new(Vec::new()),
            };
            e.build(&memory, family)?;
            if std::env::var_os("LAB8_GPU_PROFILE").is_some() {
                let period = e.instance.get_physical_device_properties(pd).limits.timestamp_period as f64;
                let pool = e
                    .device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(512),
                        None,
                    )
                    .map_err(vk_err("horodatages"))?;
                e.profile = Some((pool, period, Mutex::new([(0.0, 0); 7])));
            }
            Ok(e)
        }
    }

    pub fn describe(&self) -> String {
        format!("GPU {} (Vulkan, bf16, matrices coopératives), lots de {} positions au plus", self.name, self.max)
    }

    /// Un tampon et sa mémoire, du premier type qui a toutes les propriétés d'une des listes `wanted`.
    unsafe fn buffer(
        &mut self,
        memory: &vk::PhysicalDeviceMemoryProperties,
        size: usize,
        usage: vk::BufferUsageFlags,
        wanted: &[vk::MemoryPropertyFlags],
    ) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        // SAFETY : voir `new`
        unsafe {
            let d = &self.device;
            let buffer = d
                .create_buffer(&vk::BufferCreateInfo::default().size(size.max(4) as u64).usage(usage), None)
                .map_err(vk_err("tampon"))?;
            let req = d.get_buffer_memory_requirements(buffer);
            let types = &memory.memory_types[..memory.memory_type_count as usize];
            let index = wanted
                .iter()
                .find_map(|&flags| {
                    types
                        .iter()
                        .enumerate()
                        .position(|(i, t)| req.memory_type_bits & (1 << i) != 0 && t.property_flags.contains(flags))
                })
                .ok_or("aucune mémoire qui convienne")?;
            let mem = d
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(index as u32),
                    None,
                )
                .map_err(vk_err("mémoire"))?;
            d.bind_buffer_memory(buffer, mem, 0).map_err(vk_err("mémoire du tampon"))?;
            self.buffers.push((buffer, mem));
            Ok((buffer, mem))
        }
    }

    /// Un tampon de la mémoire du GPU rempli depuis l'hôte (les poids).
    unsafe fn upload(
        &mut self,
        memory: &vk::PhysicalDeviceMemoryProperties,
        bytes: &[u8],
    ) -> Result<vk::Buffer, String> {
        use vk::MemoryPropertyFlags as M;
        // SAFETY : voir `new`
        unsafe {
            let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
            let host = M::HOST_VISIBLE | M::HOST_COHERENT;
            let (buf, _) = self.buffer(
                memory,
                bytes.len(),
                storage | vk::BufferUsageFlags::TRANSFER_DST,
                &[M::DEVICE_LOCAL, M::empty()],
            )?;
            let (staging, smem) = self.buffer(memory, bytes.len(), vk::BufferUsageFlags::TRANSFER_SRC, &[host])?;
            let p = self
                .device
                .map_memory(smem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(vk_err("projection"))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p as *mut u8, bytes.len());
            self.device.unmap_memory(smem);
            let d = &self.device;
            d.begin_command_buffer(self.command, &vk::CommandBufferBeginInfo::default())
                .map_err(vk_err("commandes"))?;
            d.cmd_copy_buffer(self.command, staging, buf, &[vk::BufferCopy::default().size(bytes.len().max(4) as u64)]);
            d.end_command_buffer(self.command).map_err(vk_err("commandes"))?;
            self.submit_and_wait()?;
            Ok(buf)
        }
    }

    unsafe fn submit_and_wait(&self) -> Result<(), String> {
        // SAFETY : voir `new`
        unsafe {
            let d = &self.device;
            let cmds = [self.command];
            d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], self.fence)
                .map_err(vk_err("soumission"))?;
            d.wait_for_fences(&[self.fence], true, u64::MAX).map_err(vk_err("attente"))?;
            d.reset_fences(&[self.fence]).map_err(vk_err("barrière"))?;
            Ok(())
        }
    }

    unsafe fn build(&mut self, memory: &vk::PhysicalDeviceMemoryProperties, family: u32) -> Result<(), String> {
        use vk::MemoryPropertyFlags as M;
        // SAFETY : voir `new`
        unsafe {
            let d = self.device.clone();
            self.command_pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(vk_err("commandes"))?;
            self.command = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(self.command_pool).command_buffer_count(1),
                )
                .map_err(vk_err("commandes"))?[0];
            self.fence = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vk_err("barrière"))?;
            let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
            let host = M::HOST_VISIBLE | M::HOST_COHERENT;
            let wbytes: Vec<u8> = self.weights.w.iter().flat_map(|x| x.to_le_bytes()).collect();
            let wbuf = self.upload(memory, &wbytes)?;
            let w16bytes: Vec<u8> = self.weights.wb.iter().flat_map(|x| x.to_le_bytes()).collect();
            let w16buf = self.upload(memory, &w16bytes)?;
            let (abuf, _) = self.buffer(memory, self.acts.size * 4, storage, &[M::DEVICE_LOCAL])?;
            let (ibuf, imem) = self.buffer(memory, self.max * POS_BYTES, storage, &[M::DEVICE_LOCAL | host, host])?;
            let (obuf, omem) =
                self.buffer(memory, self.max * self.per16 * 2, storage, &[host | M::HOST_CACHED, host])?;
            let (fbuf, fmem) = self.buffer(memory, self.max * 20 * 4, storage, &[host | M::HOST_CACHED, host])?;
            let map = |m| d.map_memory(m, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty());
            self.input = map(imem).map_err(vk_err("projection"))? as *mut u8;
            self.out16 = map(omem).map_err(vk_err("projection"))? as *const u16;
            self.outf = map(fmem).map_err(vk_err("projection"))? as *const f32;

            let bindings: Vec<_> = (0..6)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            self.set_layout = d
                .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
                .map_err(vk_err("disposition"))?;
            let push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(128)];
            let set_layouts = [self.set_layout];
            self.layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&push),
                    None,
                )
                .map_err(vk_err("disposition"))?;
            let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(6)];
            self.descriptor_pool = d
                .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)
                .map_err(vk_err("réserve"))?;
            self.set = d
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&set_layouts),
                )
                .map_err(vk_err("descripteurs"))?[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = [wbuf, w16buf, abuf, ibuf, obuf, fbuf]
                .iter()
                .map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)])
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            d.update_descriptor_sets(&writes, &[]);
            for code in SPV {
                let words =
                    ash::util::read_spv(&mut std::io::Cursor::new(code)).map_err(|e| format!("SPIR-V : {e}"))?;
                let module = d
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                    .map_err(vk_err("shader"))?;
                self.modules.push(module);
                // des sous-groupes de 32 fils (matrices coopératives, `layernorm`)
                let mut size =
                    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main")
                    .push_next(&mut size);
                let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.layout);
                let pipeline = d
                    .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                    .map(|p| p[0])
                    .map_err(|(_, e)| format!("Vulkan, pipeline : {e}"))?;
                self.pipelines.push(pipeline);
            }
            Ok(())
        }
    }

    /// Enregistre un kernel : ses constantes, ses groupes, puis une barrière.
    unsafe fn dispatch(&self, kernel: usize, push: &[u32], groups: (u32, u32)) {
        // SAFETY : commande en cours d'enregistrement (`record`)
        unsafe {
            let d = &self.device;
            let c = self.command;
            d.cmd_bind_pipeline(c, vk::PipelineBindPoint::COMPUTE, self.pipelines[kernel]);
            let bytes: Vec<u8> = push.iter().flat_map(|x| x.to_le_bytes()).collect();
            d.cmd_push_constants(c, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            d.cmd_dispatch(c, groups.0, groups.1, 1);
            if let Some((pool, ..)) = &self.profile {
                let mut marks = self.marks.lock().expect("horodatages");
                d.cmd_write_timestamp(c, vk::PipelineStageFlags::COMPUTE_SHADER, *pool, marks.len() as u32 + 1);
                marks.push(kernel);
            }
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            d.cmd_pipeline_barrier(
                c,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// `rows` lignes : c = act(a · lin + biais) (+ c). Lignes de a : (m / ag) × ags + m % ag (de même
    /// pour c) ; `mode` comme `gemm.comp`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(&self, rows: usize, lin: Lin, a: (u32, u32, u32, u32), c: (u32, u32, u32, u32), mode: u32) {
        let push = [rows as u32, lin.n, lin.k, a.0, a.1, a.2, a.3, c.0, c.1, c.2, c.3, lin.w, lin.b, mode];
        // SAFETY : voir `dispatch`
        unsafe { self.dispatch(GEMM, &push, (rows.div_ceil(128) as u32, lin.n.div_ceil(128))) }
    }

    /// Enregistre le calcul de `b` positions.
    unsafe fn record(&self, b: usize) -> Result<(), String> {
        let w = &self.weights;
        let a = &self.acts;
        let d = w.d as u32;
        let one = u32::MAX; // pas de saut de lignes
        let t = TOKENS as u32;
        // SAFETY : voir `new`
        unsafe {
            let dev = &self.device;
            dev.reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
                .map_err(vk_err("commandes"))?;
            dev.begin_command_buffer(self.command, &vk::CommandBufferBeginInfo::default())
                .map_err(vk_err("commandes"))?;
            dev.cmd_bind_descriptor_sets(
                self.command,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
            if let Some((pool, ..)) = &self.profile {
                dev.cmd_reset_query_pool(self.command, *pool, 0, 512);
                dev.cmd_write_timestamp(self.command, vk::PipelineStageFlags::TOP_OF_PIPE, *pool, 0);
                self.marks.lock().expect("horodatages").clear();
            }
            let bu = b as u32;
            self.dispatch(
                EMBED,
                &[
                    a.region,
                    a.player,
                    a.market,
                    a.game,
                    w.scales[0],
                    w.scales[1],
                    w.scales[2],
                    w.scales[3],
                    w.player_index,
                ],
                (bu, 1),
            );
            // les entrées des jetons, chacune à sa place dans X
            let r = REGIONS as u32;
            self.gemm(b * REGIONS, w.inputs[0], (a.region, 80, one, 0), (a.x, d, r, t), 0);
            self.gemm(b * PLAYERS, w.inputs[1], (a.player, 128, one, 0), (a.x + r * d, d, PLAYERS as u32, t), 0);
            self.gemm(b * 6, w.inputs[2], (a.market, 48, one, 0), (a.x + 53 * d, d, 6, t), 0);
            self.gemm(b, w.inputs[3], (a.game, 80, one, 0), (a.x + 59 * d, d, 1, t), 0);
            self.dispatch(FIXUP, &[a.x, d, w.player_pos, w.market_pos], (bu, 1));
            let rows = b * TOKENS;
            let norm_groups = (rows.div_ceil(8) as u32, 1);
            for l in &w.layers {
                self.dispatch(LAYERNORM, &[rows as u32, d, a.x, a.n, l.ln1.g, l.ln1.b], norm_groups);
                self.gemm(rows, l.qkv, (a.n, d, one, 0), (a.qkv, 3 * d, one, 0), 0);
                self.dispatch(ATTENTION, &[a.qkv, a.att, d, w.heads as u32, l.rel], (bu, w.heads as u32));
                self.gemm(rows, l.o, (a.att, d, one, 0), (a.x, d, one, 0), 4);
                self.dispatch(LAYERNORM, &[rows as u32, d, a.x, a.n, l.ln2.g, l.ln2.b], norm_groups);
                self.gemm(rows, l.a, (a.n, d, one, 0), (a.h, 4 * d, one, 0), 1);
                self.gemm(rows, l.b, (a.h, 4 * d, one, 0), (a.x, d, one, 0), 4);
            }
            self.dispatch(LAYERNORM, &[rows as u32, d, a.x, a.o, w.enc_ln.g, w.enc_ln.b], norm_groups);
            self.dispatch(POOL, &[a.o, a.pool, d], (bu, 1));
            self.gemm(b, w.val.0, (a.pool, 2 * d, one, 0), (a.v1, d, one, 0), 2);
            self.gemm(b, w.val.1, (a.v1, d, one, 0), (a.v2, 16, one, 0), 0);
            self.gemm(b, w.next.0, (a.pool, 2 * d, one, 0), (a.n1, d, one, 0), 2);
            self.gemm(b, w.next.1, (a.n1, d, one, 0), (a.n2, 16, one, 0), 0);
            let kvw = 2 * w.dec as u32 * d;
            self.gemm(rows, w.kv, (a.o, d, one, 0), (a.kv, kvw, one, 0), 0);
            self.gemm(b * REGIONS, w.reg, (a.o, d, r, t), (a.reg, d, one, 0), 0);
            self.dispatch(PACK, &[a.o, a.reg, a.kv, a.v2, a.n2, d, w.dec as u32, self.per16 as u32], (bu, 1));
            dev.end_command_buffer(self.command).map_err(vk_err("commandes"))?;
        }
        Ok(())
    }

    /// Les lectures des positions `items` (encodage, distances de la carte).
    pub fn encode(&self, items: &[(&Encoded, &[u8])]) -> Result<Vec<Memory>, String> {
        let mut out = Vec::with_capacity(items.len());
        for chunk in items.chunks(self.max) {
            out.extend(self.encode_batch(chunk)?);
        }
        Ok(out)
    }

    fn encode_batch(&self, items: &[(&Encoded, &[u8])]) -> Result<Vec<Memory>, String> {
        let _held = self.lock.lock().expect("GPU");
        let b = items.len();
        // SAFETY : la mémoire projetée a `max` positions ; un seul lot à la fois (`lock`)
        unsafe {
            for (i, (e, dist)) in items.iter().enumerate() {
                let p = self.input.add(i * POS_BYTES);
                let bytes = std::slice::from_raw_parts_mut(p, POS_BYTES);
                for (r, row) in e.node.iter().enumerate() {
                    bytes[r * NODE..(r + 1) * NODE].copy_from_slice(row);
                }
                let g = REGIONS * NODE;
                bytes[g..g + GLOBAL].copy_from_slice(&e.global);
                bytes[g + GLOBAL..g + GLOBAL + REGIONS * REGIONS].copy_from_slice(&dist[..REGIONS * REGIONS]);
                let at = g + GLOBAL + REGIONS * REGIONS;
                bytes[at..at + 4].copy_from_slice(&[e.regions as u8, e.players as u8, 0, 0]);
            }
            let t0 = std::time::Instant::now();
            self.record(b)?;
            self.submit_and_wait()?;
            let t1 = std::time::Instant::now();
            if let Some((pool, period, totals)) = &self.profile {
                let marks = self.marks.lock().expect("horodatages");
                let mut stamps = vec![0u64; marks.len() + 1];
                self.device
                    .get_query_pool_results(
                        *pool,
                        0,
                        &mut stamps,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .map_err(vk_err("horodatages"))?;
                let mut totals = totals.lock().expect("horodatages");
                for (i, &k) in marks.iter().enumerate() {
                    totals[k].0 += (stamps[i + 1] - stamps[i]) as f64 * period;
                    totals[k].1 += 1;
                }
            }
            let d = self.weights.d;
            let mut out = Vec::with_capacity(b);
            for (i, (e, _)) in items.iter().enumerate() {
                let o16 = std::slice::from_raw_parts(self.out16.add(i * self.per16), self.per16);
                let f = std::slice::from_raw_parts(self.outf.add(i * 20), 20);
                let conv = |s: &[u16]| s.iter().map(|&h| from_bf16(h)).collect::<Vec<f32>>();
                let x = conv(&o16[..e.regions * d]);
                let reg = conv(&o16[REGIONS * d..REGIONS * d + e.regions * d]);
                let base = 2 * REGIONS * d;
                let cross = (0..self.weights.dec)
                    .map(|l| {
                        let k = base + 2 * l * TOKENS * d;
                        (conv(&o16[k..k + TOKENS * d]), conv(&o16[k + TOKENS * d..k + 2 * TOKENS * d]))
                    })
                    .collect();
                let mut heads = [0.0; 20];
                heads.copy_from_slice(f);
                out.push(Memory::from_parts(e, d, x, reg, cross, heads));
            }
            if std::env::var_os("LAB8_GPU_TIMES").is_some() {
                eprintln!(
                    "lot de {b} : GPU {:.2} ms, sorties {:.2} ms",
                    (t1 - t0).as_secs_f64() * 1e3,
                    t1.elapsed().as_secs_f64() * 1e3
                );
            }
            Ok(out)
        }
    }
}

impl Encoder {
    /// La durée cumulée de chaque sorte de kernel (`LAB8_GPU_PROFILE`).
    pub fn profile(&self) -> Option<String> {
        let (_, _, totals) = self.profile.as_ref()?;
        let totals = totals.lock().expect("horodatages");
        let all: f64 = totals.iter().map(|t| t.0).sum();
        let names = ["embed", "gemm", "fixup", "layernorm", "attention", "pool", "pack"];
        Some(
            names
                .iter()
                .zip(totals.iter())
                .map(|(n, t)| format!("{n} {:.1} % ({:.2} ms)", 100.0 * t.0 / all, t.0 / 1e6))
                .collect::<Vec<_>>()
                .join(", "),
        )
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY : plus aucun calcul en vol (`encode` attend chaque lot)
        unsafe {
            let d = &self.device;
            let _ = d.device_wait_idle();
            for &p in &self.pipelines {
                d.destroy_pipeline(p, None);
            }
            for &m in &self.modules {
                d.destroy_shader_module(m, None);
            }
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_pool(self.descriptor_pool, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            d.destroy_fence(self.fence, None);
            if let Some((pool, ..)) = &self.profile {
                d.destroy_query_pool(*pool, None);
            }
            d.destroy_command_pool(self.command_pool, None);
            for &(b, m) in &self.buffers {
                d.destroy_buffer(b, None);
                d.free_memory(m, None);
            }
            d.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// L'encodeur partagé par plusieurs fils : les positions demandées en même temps partent dans un seul
/// lot. Le premier fil qui trouve le GPU libre emporte toutes les demandes en attente (jusqu'à `max`
/// positions), les calcule, et rend à chacun les siennes.
pub struct Shared {
    pub encoder: Encoder,
    queue: Mutex<Queue>,
    ready: std::sync::Condvar,
}

#[derive(Default)]
struct Queue {
    pending: Vec<(u64, Items)>,
    done: HashMap<u64, Vec<Memory>>,
    running: bool,
    next: u64,
}

impl Shared {
    pub fn new(encoder: Encoder) -> Shared {
        Shared { encoder, queue: Mutex::default(), ready: std::sync::Condvar::new() }
    }

    pub fn encode(&self, items: Vec<(Encoded, Vec<u8>)>) -> Vec<Memory> {
        if items.is_empty() {
            return Vec::new();
        }
        let mut q = self.queue.lock().expect("file du GPU");
        let id = q.next;
        q.next += 1;
        q.pending.push((id, items));
        loop {
            if let Some(out) = q.done.remove(&id) {
                return out;
            }
            if q.running {
                q = self.ready.wait(q).expect("file du GPU");
                continue;
            }
            // ce fil calcule pour tous ceux qui attendent
            q.running = true;
            let mut taken = Vec::new();
            let mut count = 0;
            while let Some(first) = q.pending.first() {
                if count > 0 && count + first.1.len() > self.encoder.max {
                    break;
                }
                count += first.1.len();
                taken.push(q.pending.remove(0));
            }
            drop(q);
            let flat: Vec<(&Encoded, &[u8])> =
                taken.iter().flat_map(|(_, v)| v.iter().map(|(e, d)| (e, &d[..]))).collect();
            let mut out = self.encoder.encode(&flat).expect("calcul sur le GPU").into_iter();
            q = self.queue.lock().expect("file du GPU");
            for (gid, v) in &taken {
                q.done.insert(*gid, out.by_ref().take(v.len()).collect());
            }
            q.running = false;
            self.ready.notify_all();
        }
    }
}
