//! Le réseau sur GPU par Vulkan (`lab14 selfplay --device vulkan`, inclus à la compilation par
//! défaut), sans Python : le calcul de `net.rs` en un shader (`lab14/shaders/net.comp`,
//! compilé par `glslc` à la construction), sur n'importe quel GPU qui a Vulkan (AMD, NVIDIA ;
//! llvmpipe, le Vulkan de Mesa sur processeur, pour les tests sans GPU).
//!
//! Le flot (`gpu::Gpu`) : les fils de jeu écrivent leurs positions dans un anneau de la mémoire que
//! le GPU lit directement (`RING` positions ; mémoire du GPU visible de l'hôte, sinon mémoire de
//! l'hôte visible du GPU) ; un fil soumet les calculs des positions écrites (`Engine::submit`) :
//! un groupe de travail par position, tout le réseau dans la mémoire partagée du groupe ; le GPU
//! écrit les sorties dans un anneau que les fils de jeu lisent sur place.
//!
//! Calcul en f32 ; les sommes ne se font pas dans le même ordre que sur le processeur (écarts de
//! l'ordre de 1e-6).
//!
//! `LAB14_VULKAN_PROFILE=1` : la durée de chaque lot sur le GPU (horodatages), totalisée et affichée à
//! la fin.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use ash::vk;

use crate::encode::{GLOBAL, NODE, REGIONS};
use crate::moves::{GLOBAL_KEYS, KEYS};
use crate::net::{Eval, Position, REGION_KINDS};

/// Places de l'anneau des entrées et de celui des sorties (`gpu::Gpu`) : toutes les positions en
/// vol, écrites, en calcul ou pas encore lues.
pub const RING: usize = 16384;
/// Numéros de la file d'ordre : la place de chaque position à calculer, dans l'ordre des envois.
pub const ORDER: usize = 65536;
/// Calculs en vol au plus (commandes, horodatages).
pub const COMMANDS: usize = 4;
/// Le flot par étapes (`Engine::launch`) : `ROUNDS` commandes enregistrées une fois, de `STEPS`
/// étapes chacune ; une étape attend son feu vert (un événement posé par l'hôte), calcule les
/// positions que dit le tampon des étapes, puis signale sa fin (un autre événement). Une seule
/// soumission pour `STEPS` calculs.
pub const STEPS: usize = 32;
pub const ROUNDS: usize = 3;
pub const STEP_SLOTS: usize = STEPS * ROUNDS;
/// Files de calcul au plus : les lots de files différentes peuvent se chevaucher sur le GPU. Sur
/// RADV, deux files ralentissent (836 positions/s au lieu de 1 222, 4 emplacements).
const QUEUES: u32 = 1;
/// Positions au plus par calcul.
pub const MAX_BATCH: usize = 2048;
/// Nombres rendus par position : les logits des coups, puis la tête de valeur.
const OUT: usize = KEYS + 10;
/// La table en tête des poids (`net.comp`, `net32.comp`) : 16 entrées, 8 par couche (8 au plus),
/// puis les poids en bf16 : `node_in` et `node_pol` (parties haute et basse), puis 2 par couche.
const TABLE: usize = 100;
const TABLE16: usize = 80;

const NET16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/net.spv"));
const NET32_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/net32.spv"));
/// `net.comp` instrumenté (`LAB14_VULKAN_PHASES`) : la durée de chaque étape, par position.
const NET16_PROF_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/net_prof.spv"));
/// Les étapes de `net.comp` (`MARK`), dans l'ordre ; les couches cumulées.
pub const PHASES: [&str; 15] = [
    "entrées",
    "glob_in",
    "node_in",
    "voisines",
    "part de g (n1)",
    "n1",
    "n2",
    "pooling",
    "g1",
    "g2",
    "politique (partie)",
    "politique (régions)",
    "valeur",
    "n1 : calcul du 1er sous-groupe",
    "n1 : attente des autres",
];
/// Entrées d'une région complétées à un multiple de 16 (`net.comp`).
const NODE16: usize = 96;

/// `VK_KHR_shader_bfloat16` (absent d'ash 0.38) : `VkPhysicalDeviceShaderBfloat16FeaturesKHR`.
#[repr(C)]
struct Bf16Features {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    shader_bfloat16_type: vk::Bool32,
    shader_bfloat16_dot_product: vk::Bool32,
    shader_bfloat16_cooperative_matrix: vk::Bool32,
}
const BF16_FEATURES: vk::StructureType = vk::StructureType::from_raw(1_000_141_000);
const BF16_COMPONENT: vk::ComponentTypeKHR = vk::ComponentTypeKHR::from_raw(1_000_141_000);

/// f32 → bf16, au plus proche (pair en cas d'égalité).
fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Les poids pour les shaders : la table de leurs positions, puis chaque couche linéaire, poids
/// rangés entrée × sortie, et son biais ; à part, en bf16, les poids des grands produits des
/// régions (`node_in` complété à `NODE16` entrées, `n1` sans la part de g, `n2`).
struct Weights {
    data: Vec<f32>,
    data16: Vec<u16>,
    hidden: u32,
    layers: u32,
}

impl Weights {
    /// Lit un fichier de poids SMNN1 (`net.rs`).
    fn load(path: &Path) -> Result<Weights, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{} : {e}", path.display()))?;
        let mut f = &bytes[..];
        let mut take = |n: usize| -> Result<&[u8], String> {
            if f.len() < n {
                return Err("fichier de poids tronqué".into());
            }
            let (head, rest) = f.split_at(n);
            f = rest;
            Ok(head)
        };
        if take(5)? != b"SMNN1" {
            return Err("pas un fichier de poids SMNN1".into());
        }
        let u32_at = |b: &[u8]| u32::from_le_bytes(b.try_into().expect("4 octets"));
        let count = u32_at(take(4)?);
        let mut tensors = HashMap::new();
        for _ in 0..count {
            let len = u32_at(take(4)?) as usize;
            let name = String::from_utf8_lossy(take(len)?).into_owned();
            let ndim = u32_at(take(4)?) as usize;
            let dims = (0..ndim).map(|_| take(4).map(|b| u32_at(b) as usize)).collect::<Result<Vec<_>, _>>()?;
            let size: usize = dims.iter().product();
            let data: Vec<f32> = take(size * 4)?.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
            tensors.insert(name, (dims, data));
        }
        let mut data = vec![0.0f32; TABLE];
        let mut data16: Vec<u16> = Vec::new();
        // des poids en bf16 : leur position dans la table, les `rows` premières lignes de `wt`
        // les `rows` premières entrées de `wt` (entrée × sortie, `out` sorties, multiple de 16),
        // complétées à `pad`, en tuiles de 16 × 16 (`TILE` de net.comp) : tuile par tuile, les
        // tuiles d'entrées d'une même colonne de sorties à la suite, chaque tuile par colonnes
        let mut put16 = |data: &mut Vec<f32>, entry: usize, wt: &[f32], rows: usize, pad: usize, out: usize| {
            data16.resize(data16.len().next_multiple_of(64), 0);
            data[entry] = f32::from_bits(data16.len() as u32);
            for nb in 0..out / 16 {
                for kb in 0..pad / 16 {
                    for n in 0..16 {
                        for k in kb * 16..kb * 16 + 16 {
                            data16.push(if k < rows { bf16(wt[k * out + nb * 16 + n]) } else { 0 });
                        }
                    }
                }
            }
        };
        // la partie basse : ce que l'arrondi en bf16 a laissé
        let low = |wt: &[f32]| -> Vec<f32> { wt.iter().map(|&x| x - f32::from_bits((bf16(x) as u32) << 16)).collect() };
        let mut wts: HashMap<String, Vec<f32>> = HashMap::new();
        let table = |data: &mut Vec<f32>, entry: usize, v: &[f32]| {
            data[entry] = f32::from_bits(data.len() as u32);
            data.extend_from_slice(v);
        };
        // une couche linéaire : (sorties, entrées)
        let mut lin = |data: &mut Vec<f32>, entry: usize, name: &str| -> Result<(usize, usize), String> {
            let missing = || format!("poids manquants : {name}");
            let (dims, weight) = tensors.get(&format!("{name}.weight")).ok_or_else(missing)?;
            let (_, bias) = tensors.get(&format!("{name}.bias")).ok_or_else(missing)?;
            let (out, inp) = (dims[0], dims[1]);
            let mut wt = vec![0.0; inp * out];
            for o in 0..out {
                for i in 0..inp {
                    wt[i * out + o] = weight[o * inp + i];
                }
            }
            table(data, entry, &wt);
            table(data, entry + 1, bias);
            wts.insert(name.to_string(), wt);
            Ok((out, inp))
        };
        let (hidden, node) = lin(&mut data, 2, "node_in")?;
        let (_, global) = lin(&mut data, 4, "glob_in")?;
        lin(&mut data, 6, "glob_pol1")?;
        let (global_keys, _) = lin(&mut data, 8, "glob_pol2")?;
        let (kinds, _) = lin(&mut data, 10, "node_pol")?;
        lin(&mut data, 12, "val1")?;
        lin(&mut data, 14, "val2")?;
        let mut layers = 0;
        while tensors.contains_key(&format!("layers.{layers}.n1.weight")) {
            if 16 + 8 * (layers + 1) > TABLE16 {
                return Err("trop de couches".into());
            }
            for (k, part) in ["n1", "n2", "g1", "g2"].iter().enumerate() {
                lin(&mut data, 16 + 8 * layers + 2 * k, &format!("layers.{layers}.{part}"))?;
            }
            layers += 1;
        }
        let h = hidden;
        put16(&mut data, TABLE16, &wts["node_in"], NODE, NODE16, h);
        put16(&mut data, TABLE16 + 1, &low(&wts["node_in"]), NODE, NODE16, h);
        // node_pol : la part des régions (h premières entrées), complétée à 16 sorties
        let pol: Vec<f32> = (0..h * 16)
            .map(|e| if e % 16 < kinds { wts["node_pol"][(e / 16) * kinds + e % 16] } else { 0.0 })
            .collect();
        put16(&mut data, TABLE16 + 2, &pol, h, h, 16);
        put16(&mut data, TABLE16 + 3, &low(&pol), h, h, 16);
        for l in 0..layers {
            put16(&mut data, TABLE16 + 4 + 2 * l, &wts[&format!("layers.{l}.n1")], 2 * h, 2 * h, h);
            put16(&mut data, TABLE16 + 5 + 2 * l, &wts[&format!("layers.{l}.n2")], h, h, h);
        }
        for (entry, name, size) in [(0, "node_scale", NODE), (1, "global_scale", GLOBAL)] {
            let (_, v) = tensors.get(name).ok_or(format!("poids manquants : {name}"))?;
            if v.len() != size {
                return Err(format!("{name} : mauvaise taille"));
            }
            table(&mut data, entry, v);
        }
        if node != NODE || global != GLOBAL || global_keys != GLOBAL_KEYS || kinds != REGION_KINDS {
            return Err("réseau d'une autre version des entrées".into());
        }
        if hidden != 64 && hidden != 128 {
            return Err(format!("largeur {hidden} : 64 ou 128 seulement (net.comp)"));
        }
        Ok(Weights { data, data16, hidden: hidden as u32, layers: layers as u32 })
    }
}

/// Réseaux au plus d'un même moteur (`Engine::new_nets`).
pub const NETS: usize = 2;

/// Où commencent les poids d'un réseau dans les tampons (f32, bf16), et ses couches.
type Base = (u32, u32, u32);

impl Weights {
    /// Les poids de plusieurs réseaux de même largeur, à la suite dans les mêmes tampons, où
    /// commencent ceux de chacun (f32, bf16) et son nombre de couches : la table de chaque réseau
    /// donne ses positions à partir de là (`T` des shaders, décalé par réseau).
    fn several(nets: &[&Path]) -> Result<(Weights, Vec<Base>), String> {
        if nets.is_empty() || nets.len() > NETS {
            return Err(format!("de 1 à {NETS} réseaux par moteur"));
        }
        let (mut data, mut data16, mut bases) = (Vec::new(), Vec::new(), Vec::new());
        let mut hidden = None;
        for net in nets {
            let w = Weights::load(net)?;
            if hidden.is_some_and(|h| h != w.hidden) {
                return Err(format!("{} : pas la même largeur que le premier réseau", net.display()));
            }
            hidden = Some(w.hidden);
            // des débuts alignés comme dans un réseau seul (tuiles bf16 de 64 nombres)
            data.resize(data.len().next_multiple_of(64), 0.0);
            data16.resize(data16.len().next_multiple_of(64), 0);
            bases.push((data.len() as u32, data16.len() as u32, w.layers));
            data.extend_from_slice(&w.data);
            data16.extend_from_slice(&w.data16);
        }
        let layers = bases[0].2;
        Ok((Weights { data, data16, hidden: hidden.expect("un réseau"), layers }, bases))
    }
}

/// Où sont les entrées des positions dans l'anneau (octets), et la file d'ordre (`ORDER` numéros
/// de places).
#[derive(Clone, Copy)]
struct InputLayout {
    node: usize,
    glob: usize,
    adj: usize,
    regions: usize,
    /// le réseau de chaque position (un octet, `Engine::new_nets`)
    nets: usize,
    order: usize,
    size: usize,
}

impl InputLayout {
    fn new(max: usize) -> InputLayout {
        let up = |x: usize| x.next_multiple_of(16);
        let node = 0;
        let glob = up(node + max * REGIONS * NODE);
        let adj = up(glob + max * GLOBAL);
        let regions = up(adj + max * REGIONS * 8);
        let nets = up(regions + max);
        let order = up(nets + max);
        InputLayout { node, glob, adj, regions, nets, order, size: order + ORDER * 4 }
    }
}

/// Le profil (`LAB14_VULKAN_PROFILE`) : un horodatage au début et à la fin de chaque calcul.
struct Profile {
    queries: vk::QueryPool,
    /// Nanosecondes par unité d'horodatage.
    period: f64,
    /// Lots, positions, durée totale sur le GPU (ns).
    totals: Mutex<(u64, u64, f64)>,
    /// Avec `LAB14_LATENCY` : de quoi ramener les horodatages du GPU à l'horloge `CLOCK_MONOTONIC`
    /// (`profile_lot`).
    calibrated: Option<ash::khr::calibrated_timestamps::Device>,
}

/// Le réseau sur un GPU, avec un anneau de `RING` positions (entrées et sorties).
pub struct Engine {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    /// Les files de calcul ; seule la première sert aux calculs.
    queues: Vec<vk::Queue>,
    command_pool: vk::CommandPool,
    commands: Vec<vk::CommandBuffer>,
    /// Les commandes des étapes (`ROUNDS`), le tampon des étapes (par étape : groupes x, y, z, puis
    /// le premier numéro de la file d'ordre), les feux verts et les fins.
    rounds: Vec<vk::CommandBuffer>,
    steps_buf: vk::Buffer,
    steps: *mut u32,
    go: Vec<vk::Event>,
    fin: Vec<vk::Event>,
    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    module: vk::ShaderModule,
    buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    input: *mut u8,
    output: *const f32,
    /// Un sémaphore à compteur : il passe à `n` à la fin du `n`-ième calcul.
    timeline: vk::Semaphore,
    /// Le nombre de calculs soumis : un seul fil soumet à la fois.
    submitted: Mutex<u64>,
    profile: Option<Profile>,
    weights: Weights,
    /// où commencent les poids de chaque réseau (f32, bf16), et ses couches
    bases: Vec<Base>,
    input_layout: InputLayout,
    max: usize,
    name: String,
    /// Les grands produits en bf16 sur les matrices coopératives (`net.comp`) ; sinon tout en f32
    /// (`net32.comp`).
    bf16: bool,
    /// Le shader instrumenté (`LAB14_VULKAN_PHASES`) : ses durées par étape, 16 par position, dans
    /// `prof` (mémoire projetée).
    phases: bool,
    prof: *mut u32,
}

// SAFETY : les pointeurs sont ceux de la mémoire projetée (`prof` : lu seulement par `bench`), qui
// vit autant que l'`Engine` ; chaque position de l'anneau n'est touchée que par qui la tient
// (`gpu::Gpu`), la file et les commandes sous `submitted`.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

fn vk_err(what: &str) -> impl Fn(vk::Result) -> String + '_ {
    move |e| format!("Vulkan, {what} : {e}")
}

impl Engine {
    /// Le GPU (le premier GPU dédié, sinon intégré, sinon n'importe quel appareil Vulkan ; ou celui
    /// dont le nom contient `LAB14_VULKAN_DEVICE`) et les poids `net`, pour des lots de `max`
    /// positions au plus.
    pub fn new(net: &Path, max: usize) -> Result<Engine, String> {
        Engine::new_nets(&[net], max)
    }

    /// `new` avec plusieurs réseaux de même largeur (`NETS` au plus) : chaque position dit le sien
    /// (`write`). Un seul moteur pour tous : deux moteurs à la fois sur ce GPU se bloquent (leurs
    /// étapes soumises d'avance attendent chacune leur feu vert, journal).
    pub fn new_nets(nets: &[&Path], max: usize) -> Result<Engine, String> {
        let (weights, bases) = Weights::several(nets)?;
        // SAFETY : appels Vulkan dans l'ordre de la spécification ; chaque objet créé est détruit
        // par `Drop`
        unsafe {
            // RADV 26.2, gfx1201 : ACO forme des paires d'instructions VOPD qui lisent un registre
            // avant qu'il ne reçoive la valeur attendue (conversions bf16 → f32 groupées : sommes
            // fausses, journal du 23 octobre) ; sans ces paires, même vitesse à 1 % près. Lu par le
            // pilote à la création de l'appareil ; `LAB14_VULKAN_VOPD=1` les garde.
            if std::env::var_os("LAB14_VULKAN_VOPD").is_none() {
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
            let wanted = std::env::var("LAB14_VULKAN_DEVICE").ok();
            let mut best: Option<(u32, vk::PhysicalDevice, u32, u32, String, f32)> = None;
            for pd in instance.enumerate_physical_devices().map_err(vk_err("appareils"))? {
                let props = instance.get_physical_device_properties(pd);
                let name = props.device_name_as_c_str().unwrap_or(c"?").to_string_lossy().into_owned();
                let rank = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    _ => 3,
                };
                if wanted.as_ref().is_some_and(|w| !name.contains(w.as_str())) {
                    continue;
                }
                // la première famille qui calcule (la file graphique, sur RADV : aussi rapide que celle de
                // calcul seul)
                let families = instance.get_physical_device_queue_family_properties(pd);
                let compute = |q: &vk::QueueFamilyProperties| q.queue_flags.contains(vk::QueueFlags::COMPUTE);
                let family = families.iter().position(compute);
                if let Some(family) = family
                    && best.as_ref().is_none_or(|(r, ..)| rank < *r)
                {
                    let count = families[family].queue_count.min(QUEUES);
                    best = Some((rank, pd, family as u32, count, name, props.limits.timestamp_period));
                }
            }
            let Some((_, pd, family, queue_count, name, period)) = best else {
                instance.destroy_instance(None);
                return Err("aucun appareil Vulkan qui calcule".into());
            };
            let bf16 = std::env::var_os("LAB14_VULKAN_F32").is_none() && Engine::has_bf16(&entry, &instance, pd);
            let phases = bf16 && std::env::var_os("LAB14_VULKAN_PHASES").is_some();
            let priorities = vec![1.0; queue_count as usize];
            let queues =
                [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
            let mut b16 = Bf16Features {
                s_type: BF16_FEATURES,
                p_next: std::ptr::null_mut(),
                shader_bfloat16_type: vk::TRUE,
                shader_bfloat16_dot_product: vk::FALSE,
                shader_bfloat16_cooperative_matrix: vk::TRUE,
            };
            let mut clock = vk::PhysicalDeviceShaderClockFeaturesKHR::default().shader_device_clock(true);
            let mut extensions = Vec::new();
            if bf16 {
                v12 = v12.shader_float16(true).vulkan_memory_model(true);
                v13 = v13.subgroup_size_control(true);
                coop = coop.cooperative_matrix(true);
                // la chaîne, du bout vers le début : (horloge →) bf16 → coop → v13 → v12
                extensions = vec![c"VK_KHR_cooperative_matrix".as_ptr(), c"VK_KHR_shader_bfloat16".as_ptr()];
                if phases {
                    b16.p_next = &mut clock as *mut _ as *mut std::ffi::c_void;
                    extensions.push(c"VK_KHR_shader_clock".as_ptr());
                }
                coop.p_next = &mut b16 as *mut Bf16Features as *mut std::ffi::c_void;
                v13.p_next = &mut coop as *mut _ as *mut std::ffi::c_void;
            }
            let latency =
                std::env::var_os("LAB14_VULKAN_PROFILE").is_some() && std::env::var_os("LAB14_LATENCY").is_some();
            if latency {
                extensions.push(c"VK_KHR_calibrated_timestamps".as_ptr());
            }
            v12.p_next = &mut v13 as *mut _ as *mut std::ffi::c_void;
            let mut info =
                vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&extensions);
            info.p_next = &v12 as *const _ as *const std::ffi::c_void;
            let device = instance.create_device(pd, &info, None).map_err(vk_err("appareil"))?;
            let queues: Vec<vk::Queue> = (0..queue_count).map(|i| device.get_device_queue(family, i)).collect();
            let memory = instance.get_physical_device_memory_properties(pd);
            let mut engine = Engine {
                _entry: entry,
                instance,
                device,
                queues,
                command_pool: vk::CommandPool::null(),
                commands: Vec::new(),
                rounds: Vec::new(),
                steps_buf: vk::Buffer::null(),
                steps: std::ptr::null_mut(),
                go: Vec::new(),
                fin: Vec::new(),
                descriptor_pool: vk::DescriptorPool::null(),
                set_layout: vk::DescriptorSetLayout::null(),
                set: vk::DescriptorSet::null(),
                layout: vk::PipelineLayout::null(),
                pipeline: vk::Pipeline::null(),
                module: vk::ShaderModule::null(),
                buffers: Vec::new(),
                input: std::ptr::null_mut(),
                output: std::ptr::null(),
                timeline: vk::Semaphore::null(),
                submitted: Mutex::new(0),
                profile: None,
                input_layout: InputLayout::new(RING),
                weights,
                bases,
                max,
                name,
                bf16,
                phases,
                prof: std::ptr::null_mut(),
            };
            engine.build(&memory, family)?;
            if std::env::var_os("LAB14_VULKAN_PROFILE").is_some() {
                let queries = engine
                    .device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2 * (COMMANDS + STEP_SLOTS) as u32),
                        None,
                    )
                    .map_err(vk_err("horodatages"))?;
                let calibrated =
                    latency.then(|| ash::khr::calibrated_timestamps::Device::new(&engine.instance, &engine.device));
                engine.profile =
                    Some(Profile { queries, period: period as f64, totals: Mutex::new((0, 0, 0.0)), calibrated });
            }
            engine.record_rounds()?;
            Ok(engine)
        }
    }

    /// Ce qui calcule.
    pub fn describe(&self) -> String {
        let how = if self.bf16 { "bf16, matrices coopératives" } else { "f32" };
        format!("GPU {} (Vulkan, {how}, {} files)", self.name, self.queues.len())
    }

    /// L'appareil a-t-il les matrices coopératives 16 × 16 × 16 en bf16 (sommes en f32), le bf16 et
    /// des sous-groupes de 32 fils ?
    unsafe fn has_bf16(entry: &ash::Entry, instance: &ash::Instance, pd: vk::PhysicalDevice) -> bool {
        // SAFETY : appareil de cette instance
        unsafe {
            let Ok(exts) = instance.enumerate_device_extension_properties(pd) else { return false };
            let has = |name: &std::ffi::CStr| exts.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
            if !has(c"VK_KHR_cooperative_matrix") || !has(c"VK_KHR_shader_bfloat16") {
                return false;
            }
            let mut b16 = Bf16Features {
                s_type: BF16_FEATURES,
                p_next: std::ptr::null_mut(),
                shader_bfloat16_type: vk::FALSE,
                shader_bfloat16_dot_product: vk::FALSE,
                shader_bfloat16_cooperative_matrix: vk::FALSE,
            };
            let mut coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR {
                p_next: &mut b16 as *mut Bf16Features as *mut std::ffi::c_void,
                ..Default::default()
            };
            let mut v12 = vk::PhysicalDeviceVulkan12Features {
                p_next: &mut coop as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            let mut v13 = vk::PhysicalDeviceVulkan13Features {
                p_next: &mut v12 as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            let mut features = vk::PhysicalDeviceFeatures2 {
                p_next: &mut v13 as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            instance.get_physical_device_features2(pd, &mut features);
            if coop.cooperative_matrix == 0
                || b16.shader_bfloat16_type == 0
                || b16.shader_bfloat16_cooperative_matrix == 0
                || v12.shader_float16 == 0
                || v12.vulkan_memory_model == 0
                || v13.subgroup_size_control == 0
            {
                return false;
            }
            let mut v13p = vk::PhysicalDeviceVulkan13Properties::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut v13p);
            instance.get_physical_device_properties2(pd, &mut props);
            if !(v13p.min_subgroup_size..=v13p.max_subgroup_size).contains(&32)
                || !v13p.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE)
            {
                return false;
            }
            let loader = ash::khr::cooperative_matrix::Instance::new(entry, instance);
            let shapes = loader.get_physical_device_cooperative_matrix_properties(pd).unwrap_or_default();
            shapes.iter().any(|c| {
                (c.m_size, c.n_size, c.k_size) == (16, 16, 16)
                    && c.a_type == BF16_COMPONENT
                    && c.b_type == BF16_COMPONENT
                    && c.c_type == vk::ComponentTypeKHR::FLOAT32
                    && c.result_type == vk::ComponentTypeKHR::FLOAT32
                    && c.scope == vk::ScopeKHR::SUBGROUP
            })
        }
    }

    /// Un tampon et sa mémoire, du premier type qui a toutes les propriétés d'une des listes
    /// `wanted` (dans l'ordre de préférence).
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
                .create_buffer(&vk::BufferCreateInfo::default().size(size as u64).usage(usage), None)
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

    unsafe fn build(&mut self, memory: &vk::PhysicalDeviceMemoryProperties, family: u32) -> Result<(), String> {
        use vk::MemoryPropertyFlags as M;
        // SAFETY : voir `new`
        unsafe {
            let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
            let host = M::HOST_VISIBLE | M::HOST_COHERENT;
            // les poids : écrits une fois, depuis une mémoire de l'hôte
            let wbytes = self.weights.data.len() * 4;
            let (wbuf, _) = self.buffer(
                memory,
                wbytes,
                storage | vk::BufferUsageFlags::TRANSFER_DST,
                &[M::DEVICE_LOCAL, M::empty()],
            )?;
            let (staging, smem) = self.buffer(memory, wbytes, vk::BufferUsageFlags::TRANSFER_SRC, &[host])?;
            let p = self
                .device
                .map_memory(smem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(vk_err("projection"))?;
            std::ptr::copy_nonoverlapping(self.weights.data.as_ptr() as *const u8, p as *mut u8, wbytes);
            self.device.unmap_memory(smem);
            let w16bytes = (self.weights.data16.len() * 2).max(4);
            let (w16buf, _) = self.buffer(
                memory,
                w16bytes,
                storage | vk::BufferUsageFlags::TRANSFER_DST,
                &[M::DEVICE_LOCAL, M::empty()],
            )?;
            let (staging16, smem16) = self.buffer(memory, w16bytes, vk::BufferUsageFlags::TRANSFER_SRC, &[host])?;
            let p = self
                .device
                .map_memory(smem16, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(vk_err("projection"))?;
            std::ptr::copy_nonoverlapping(
                self.weights.data16.as_ptr() as *const u8,
                p as *mut u8,
                self.weights.data16.len() * 2,
            );
            self.device.unmap_memory(smem16);
            // l'anneau des entrées : écrites par l'hôte, lues par le GPU (de préférence dans sa
            // mémoire, si l'hôte la voit) ; celui des sorties : écrites par le GPU, lues par l'hôte
            // (de préférence en cache de l'hôte)
            let (ibuf, imem) = self.buffer(memory, self.input_layout.size, storage, &[M::DEVICE_LOCAL | host, host])?;
            let (obuf, omem) = self.buffer(memory, RING * OUT * 4, storage, &[host | M::HOST_CACHED, host])?;
            let map = |m| self.device.map_memory(m, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty());
            self.input = map(imem).map_err(vk_err("projection"))? as *mut u8;
            self.output = map(omem).map_err(vk_err("projection"))? as *const f32;

            // les durées par étape (`phases`) : toujours là, pour une seule disposition
            let (pbuf, pmem) = self.buffer(memory, RING * 16 * 4, storage, &[host])?;
            let pp = self
                .device
                .map_memory(pmem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(vk_err("projection"))?;
            self.prof = pp as *mut u32;
            // le tampon des étapes, écrit par l'hôte, lu par le GPU (`launch`)
            let indirect = vk::BufferUsageFlags::INDIRECT_BUFFER;
            let (sbuf, smem) = self.buffer(memory, STEP_SLOTS * 16, storage | indirect, &[host])?;
            self.steps_buf = sbuf;
            let sp = self
                .device
                .map_memory(smem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(vk_err("projection"))?;
            self.steps = sp as *mut u32;

            let d = &self.device;
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
            let push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(60)];
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
            let layouts = [self.set_layout];
            self.set = d
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&layouts),
                )
                .map_err(vk_err("descripteurs"))?[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = [wbuf, ibuf, obuf, w16buf, pbuf, sbuf]
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

            let code = match (self.bf16, self.phases) {
                (true, true) => NET16_PROF_SPV,
                (true, false) => NET16_SPV,
                _ => NET32_SPV,
            };
            // essais : un autre shader compilé (`LAB14_VULKAN_SPV=fichier.spv`, mêmes entrées et sorties)
            let other = std::env::var_os("LAB14_VULKAN_SPV")
                .map(std::fs::read)
                .transpose()
                .map_err(|e| format!("SPIR-V : {e}"))?;
            let code = other.as_deref().unwrap_or(code);
            let words = ash::util::read_spv(&mut std::io::Cursor::new(code)).map_err(|e| format!("SPIR-V : {e}"))?;
            self.module = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(vk_err("shader"))?;
            // la largeur du réseau, constante du shader
            let hidden = self.weights.hidden.to_le_bytes();
            let entries = [vk::SpecializationMapEntry::default().constant_id(0).offset(0).size(4)];
            let spec = vk::SpecializationInfo::default().map_entries(&entries).data(&hidden);
            // les matrices coopératives : des sous-groupes de 32 fils (`net.comp`)
            let mut size = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(32);
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(self.module)
                .name(c"main")
                .specialization_info(&spec);
            if self.bf16 {
                stage = stage.push_next(&mut size);
            }
            let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.layout);
            self.pipeline = d
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map(|p| p[0])
                .map_err(|(_, e)| format!("Vulkan, pipeline : {e}"))?;

            self.command_pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(vk_err("commandes"))?;
            self.commands = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(COMMANDS as u32),
                )
                .map_err(vk_err("commandes"))?;
            self.rounds = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(ROUNDS as u32),
                )
                .map_err(vk_err("commandes"))?;
            for _ in 0..STEP_SLOTS {
                let event = |d: &ash::Device| d.create_event(&vk::EventCreateInfo::default(), None);
                self.go.push(event(d).map_err(vk_err("événement"))?);
                self.fin.push(event(d).map_err(vk_err("événement"))?);
            }
            let mut kind = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
            self.timeline = d
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut kind), None)
                .map_err(vk_err("sémaphore"))?;

            // la copie des poids, une fois
            let cb = self.commands[0];
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()).map_err(vk_err("commandes"))?;
            d.cmd_copy_buffer(cb, staging, wbuf, &[vk::BufferCopy::default().size(wbytes as u64)]);
            d.cmd_copy_buffer(cb, staging16, w16buf, &[vk::BufferCopy::default().size(w16bytes as u64)]);
            d.end_command_buffer(cb).map_err(vk_err("commandes"))?;
            let cbs = [cb];
            d.queue_submit(self.queues[0], &[vk::SubmitInfo::default().command_buffers(&cbs)], vk::Fence::null())
                .map_err(vk_err("envoi"))?;
            d.queue_wait_idle(self.queues[0]).map_err(vk_err("attente"))?;
            Ok(())
        }
    }

    /// La mémoire des entrées de l'anneau.
    #[allow(clippy::mut_from_ref)]
    fn input(&self) -> &mut [u8] {
        // SAFETY : mémoire projetée de `input_layout.size` octets ; chaque position n'est écrite que
        // par qui la tient (`gpu::Gpu`)
        unsafe { std::slice::from_raw_parts_mut(self.input, self.input_layout.size) }
    }

    /// Les sorties de la position `cell` de l'anneau : `OUT` nombres.
    pub(crate) fn output(&self, cell: usize) -> &[f32] {
        // SAFETY : mémoire projetée de `RING · OUT` nombres
        unsafe { std::slice::from_raw_parts(self.output.add(cell * OUT), OUT) }
    }

    /// Écrit la position `p` à la place `cell` de l'anneau : ses régions présentes, la partie, les
    /// voisines (bits), le nombre de régions, et le réseau qui l'évalue (`new_nets`).
    pub(crate) fn write<P: Position + ?Sized>(&self, cell: usize, p: &P, net: u8) {
        let (l, input) = (self.input_layout, self.input());
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
        let o = self.input_layout.order + at * 4;
        self.input()[o..o + 4].copy_from_slice(&(cell as u32).to_le_bytes());
    }

    /// Soumet le calcul des `count` positions de la file d'ordre à partir de son numéro `first`
    /// (les suivantes reprennent au début) ; rend le numéro du calcul, que prend le sémaphore à sa
    /// fin (`wait`). `COMMANDS` calculs en vol au plus.
    pub(crate) fn submit(&self, first: usize, count: usize) -> Result<u64, String> {
        assert!(count <= self.max && first < ORDER, "calcul trop grand");
        let mut submitted = self.submitted.lock().expect("verrou");
        let value = *submitted + 1;
        let at = (value % COMMANDS as u64) as usize;
        let cb = self.commands[at];
        let params = self.params(first, u32::MAX as usize);
        // SAFETY : la commande n'est plus en vol (au plus `COMMANDS` calculs, l'appelant a attendu
        // les plus vieux) ; un seul fil soumet à la fois (`submitted`)
        unsafe {
            let d = &self.device;
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()).map_err(vk_err("commandes"))?;
            d.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(vk_err("commandes"))?;
            if let Some(p) = &self.profile {
                d.cmd_reset_query_pool(cb, p.queries, 2 * at as u32, 2);
                d.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, p.queries, 2 * at as u32);
            }
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[self.set], &[]);
            d.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &params);
            d.cmd_dispatch(cb, count as u32, 1, 1);
            if let Some(p) = &self.profile {
                d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, p.queries, 2 * at as u32 + 1);
            }
            let to_host = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &to_host,
                &[],
                &[],
            );
            d.end_command_buffer(cb).map_err(vk_err("commandes"))?;
            let values = [value];
            let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
            let (cbs, sems) = ([cb], [self.timeline]);
            let info =
                vk::SubmitInfo::default().command_buffers(&cbs).signal_semaphores(&sems).push_next(&mut timeline);
            d.queue_submit(self.queues[0], &[info], vk::Fence::null()).map_err(vk_err("envoi"))?;
        }
        *submitted = value;
        Ok(value)
    }

    /// Les constantes du shader : les entrées, les couches, puis le calcul (`first` dans la file
    /// d'ordre, ou celui que dit l'étape `step` du tampon des étapes, sinon `u32::MAX`).
    fn params(&self, first: usize, step: usize) -> Vec<u8> {
        let l = self.input_layout;
        let base = |n: usize| self.bases.get(n).or(self.bases.first()).copied().expect("un réseau");
        let (b0, b1) = (base(0), base(1));
        [l.node, l.glob, l.adj, l.regions, b0.2 as usize, first, ORDER, l.order, step, l.nets]
            .into_iter()
            .map(|v| v as u32)
            .chain([b0.0, b0.1, b1.0, b1.1, b1.2])
            .flat_map(u32::to_le_bytes)
            .collect()
    }

    /// Enregistre une fois les commandes des étapes (`ROUNDS`, voir `STEPS`).
    fn record_rounds(&mut self) -> Result<(), String> {
        // SAFETY : commandes neuves, objets vivants
        unsafe {
            let d = &self.device;
            for r in 0..ROUNDS {
                let cb = self.rounds[r];
                d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()).map_err(vk_err("commandes"))?;
                if let Some(p) = &self.profile {
                    d.cmd_reset_query_pool(cb, p.queries, 2 * (COMMANDS + r * STEPS) as u32, 2 * STEPS as u32);
                }
                for j in 0..STEPS {
                    let s = r * STEPS + j;
                    let query = 2 * (COMMANDS + s) as u32;
                    // le feu vert de l'hôte, et ce qu'il a écrit avant (positions, tampon des étapes)
                    let from_host = [vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::HOST_WRITE)
                        .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ | vk::AccessFlags::SHADER_READ)];
                    d.cmd_wait_events(
                        cb,
                        &[self.go[s]],
                        vk::PipelineStageFlags::HOST,
                        vk::PipelineStageFlags::DRAW_INDIRECT | vk::PipelineStageFlags::COMPUTE_SHADER,
                        &from_host,
                        &[],
                        &[],
                    );
                    if let Some(p) = &self.profile {
                        d.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, p.queries, query);
                    }
                    d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
                    d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[self.set], &[]);
                    d.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &self.params(0, s));
                    d.cmd_dispatch_indirect(cb, self.steps_buf, (s * 16) as u64);
                    if let Some(p) = &self.profile {
                        d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, p.queries, query + 1);
                    }
                    // les sorties pour l'hôte, puis la fin de l'étape
                    let to_host = [vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)];
                    d.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &to_host,
                        &[],
                        &[],
                    );
                    d.cmd_set_event(cb, self.fin[s], vk::PipelineStageFlags::ALL_COMMANDS);
                }
                d.end_command_buffer(cb).map_err(vk_err("commandes"))?;
            }
        }
        Ok(())
    }

    /// Soumet les commandes des étapes `round` (elles attendront leurs feux verts) ; rend le
    /// numéro de la soumission, que prend le sémaphore à la fin des `STEPS` étapes.
    pub(crate) fn submit_round(&self, round: usize) -> Result<u64, String> {
        let mut submitted = self.submitted.lock().expect("verrou");
        let value = *submitted + 1;
        let values = [value];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let (cbs, sems) = ([self.rounds[round]], [self.timeline]);
        let info = vk::SubmitInfo::default().command_buffers(&cbs).signal_semaphores(&sems).push_next(&mut timeline);
        // SAFETY : les commandes de `round` ne sont plus en vol (l'appelant a attendu leur fin)
        unsafe { self.device.queue_submit(self.queues[0], &[info], vk::Fence::null()) }.map_err(vk_err("envoi"))?;
        *submitted = value;
        Ok(value)
    }

    /// Donne le feu vert à l'étape `slot` : `count` positions de la file d'ordre à partir de `first`.
    pub(crate) fn launch(&self, slot: usize, first: usize, count: usize) -> Result<(), String> {
        assert!(count <= self.max && first < ORDER, "calcul trop grand");
        // SAFETY : mémoire projetée de `STEP_SLOTS` × 4 nombres ; l'étape n'est pas en cours
        unsafe {
            let at = self.steps.add(slot * 4);
            for (k, v) in [count as u32, 1, 1, first as u32].into_iter().enumerate() {
                at.add(k).write_volatile(v);
            }
            self.device.set_event(self.go[slot]).map_err(vk_err("feu vert"))
        }
    }

    /// L'étape `slot` est-elle finie ?
    pub(crate) fn finished(&self, slot: usize) -> Result<bool, String> {
        // SAFETY : événement vivant
        unsafe { self.device.get_event_status(self.fin[slot]) }.map_err(vk_err("fin d'étape"))
    }

    /// Remet l'étape `slot`, finie, à son état de départ.
    pub(crate) fn recycle(&self, slot: usize) -> Result<(), String> {
        // SAFETY : événements vivants, que le GPU ne touche plus avant le prochain feu vert
        unsafe {
            self.device.reset_event(self.go[slot]).map_err(vk_err("feu vert"))?;
            self.device.reset_event(self.fin[slot]).map_err(vk_err("fin d'étape"))
        }
    }

    /// Attend la fin du calcul `value`.
    pub(crate) fn wait(&self, value: u64) -> Result<(), String> {
        self.wait_for(value, u64::MAX).map(|_| ())
    }

    /// Attend la fin du calcul `value`, `timeout` ns au plus ; rend s'il est fini.
    pub(crate) fn wait_for(&self, value: u64, timeout: u64) -> Result<bool, String> {
        let (sems, values) = ([self.timeline], [value]);
        // SAFETY : sémaphore vivant ; l'attente peut se faire de n'importe quel fil
        let waited = unsafe {
            self.device.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&values), timeout)
        };
        match waited {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(vk_err("attente")(e)),
        }
    }

    /// Durée moyenne sur le GPU (ms) d'un lot de `count` positions de `regions` régions (en anneau,
    /// deux voisines chacune), sur `rounds` lots ; demande `LAB14_VULKAN_PROFILE`.
    pub fn bench(&self, count: usize, regions: usize, rounds: usize) -> Result<f64, String> {
        let l = self.input_layout;
        let input = self.input();
        input.fill(0);
        for at in 0..count {
            self.order(at, at);
        }
        for pos in 0..count {
            input[l.regions + pos] = regions as u8;
            for i in 0..regions {
                let bits = (1u64 << ((i + 1) % regions)) | (1u64 << ((i + regions - 1) % regions));
                let at = l.adj + (pos * REGIONS + i) * 8;
                input[at..at + 8].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let p = self.profile.as_ref().ok_or("LAB14_VULKAN_PROFILE=1 attendu")?;
        *p.totals.lock().expect("verrou") = (0, 0, 0.0);
        for _ in 0..rounds {
            let v = self.submit(0, count)?;
            self.wait(v)?;
            self.profile_lot(v, count);
        }
        let (lots, _, ns) = *p.totals.lock().expect("verrou");
        Ok(ns / 1e6 / lots.max(1) as f64)
    }

    /// En flux : `lots` lots de `count` positions soumis à la suite, `COMMANDS` en vol au plus ;
    /// rend la durée murale totale (ms) et la somme des durées sur le GPU (ms). Leur écart : ce que
    /// coûtent les soumissions et les fins de lot.
    pub fn bench_stream(&self, count: usize, regions: usize, lots: usize) -> Result<(f64, f64), String> {
        // les mêmes positions pour tous les lots
        self.bench(count, regions, 1)?;
        let p = self.profile.as_ref().ok_or("LAB14_VULKAN_PROFILE=1 attendu")?;
        *p.totals.lock().expect("verrou") = (0, 0, 0.0);
        let start = std::time::Instant::now();
        let mut pending = std::collections::VecDeque::new();
        for _ in 0..lots {
            if pending.len() == COMMANDS {
                let v = pending.pop_front().expect("un lot");
                self.wait(v)?;
                self.profile_lot(v, count);
            }
            pending.push_back(self.submit(0, count)?);
        }
        for v in pending {
            self.wait(v)?;
            self.profile_lot(v, count);
        }
        let wall = start.elapsed().as_secs_f64() * 1e3;
        let (_, _, ns) = *p.totals.lock().expect("verrou");
        Ok((wall, ns / 1e6))
    }

    /// Avec `LAB14_VULKAN_PHASES` : la durée moyenne de chaque étape (`PHASES`, µs, horloge du GPU à
    /// 100 MHz) d'une position, sur les `rounds` lots de `count` positions de `bench`.
    pub fn bench_phases(&self, count: usize, regions: usize, rounds: usize) -> Result<Option<Vec<f64>>, String> {
        if !self.phases {
            return Ok(None);
        }
        // SAFETY : mémoire projetée de `max · 16` nombres, que le GPU ne touche qu'entre `submit`
        // et `wait`
        let prof = unsafe { std::slice::from_raw_parts_mut(self.prof, RING * 16) };
        prof.fill(0);
        self.bench(count, regions, rounds)?;
        // chaque lot réécrit ses durées : celles du dernier
        let per = count as f64;
        Ok(Some(
            (0..PHASES.len())
                .map(|ph| (0..count).map(|pos| prof[pos * 16 + ph] as f64).sum::<f64>() / per / 100.0)
                .collect(),
        ))
    }

    /// Ajoute au profil la durée du calcul fini `value` (`count` positions).
    /// Avec `LAB14_LATENCY` : rend son début et sa fin sur le GPU, en ns de `CLOCK_MONOTONIC`.
    pub(crate) fn profile_lot(&self, value: u64, count: usize) -> Option<(u64, u64)> {
        self.profile_queries(2 * (value % COMMANDS as u64) as u32, count)
    }

    /// `profile_lot`, pour l'étape finie `slot`.
    pub(crate) fn profile_step(&self, slot: usize, count: usize) -> Option<(u64, u64)> {
        self.profile_queries(2 * (COMMANDS + slot) as u32, count)
    }

    /// Les deux horodatages à partir de `query`.
    fn profile_queries(&self, query: u32, count: usize) -> Option<(u64, u64)> {
        let p = self.profile.as_ref()?;
        let mut stamps = [0u64; 2];
        // SAFETY : le lot est fini (`wait`) ; horodatages écrits par ses commandes
        let read = unsafe {
            self.device.get_query_pool_results(
                p.queries,
                query,
                &mut stamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        };
        if read.is_ok() {
            let mut totals = p.totals.lock().expect("verrou");
            totals.0 += 1;
            totals.1 += count as u64;
            totals.2 += stamps[1].saturating_sub(stamps[0]) as f64 * p.period;
        }
        // l'horloge du GPU et celle du processeur au même instant, puis les horodatages du lot
        let c = p.calibrated.as_ref().filter(|_| read.is_ok())?;
        let infos = [vk::TimeDomainKHR::DEVICE, vk::TimeDomainKHR::CLOCK_MONOTONIC]
            .map(|d| vk::CalibratedTimestampInfoKHR::default().time_domain(d));
        // SAFETY : appareil vivant, extension activée (`latency`)
        let (now, _) = unsafe { c.get_calibrated_timestamps(&infos) }.ok()?;
        let back = |stamp: u64| now[1] - (now[0].saturating_sub(stamp) as f64 * p.period) as u64;
        Some((back(stamps[0]), back(stamps[1])))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // SAFETY : plus rien en vol après `device_wait_idle` ; chaque objet détruit une fois
        unsafe {
            let d = &self.device;
            let _ = d.device_wait_idle();
            if let Some(p) = &self.profile {
                let (lots, positions, ns) = *p.totals.lock().expect("verrou");
                eprintln!(
                    "profil Vulkan : {lots} lots, {:.0} positions par lot, {:.3} ms de GPU par lot, {:.0} positions/s de GPU",
                    positions as f64 / lots.max(1) as f64,
                    ns / 1e6 / lots.max(1) as f64,
                    positions as f64 / (ns / 1e9)
                );
                d.destroy_query_pool(p.queries, None);
            }
            d.destroy_semaphore(self.timeline, None);
            for &e in self.go.iter().chain(&self.fin) {
                d.destroy_event(e, None);
            }
            d.destroy_command_pool(self.command_pool, None);
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_shader_module(self.module, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_pool(self.descriptor_pool, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            for &(b, m) in &self.buffers {
                d.destroy_buffer(b, None);
                d.free_memory(m, None);
            }
            d.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// L'évaluation d'une position : `OUT` nombres rendus par le GPU.
pub(crate) fn eval_of(players: usize, out: &[f32]) -> Eval {
    let v: &[f32; 10] = out[KEYS..OUT].try_into().expect("10 valeurs");
    Eval::from_heads(players, out[..KEYS].to_vec(), v)
}
