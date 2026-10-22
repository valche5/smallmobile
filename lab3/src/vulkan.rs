//! Le réseau sur GPU par Vulkan (`lab3 selfplay --device vulkan`, programme compilé avec
//! `--features vulkan`), sans Python : le calcul de `net.rs` en un shader (`lab3/shaders/net.comp`,
//! compilé par `glslc` à la construction), sur n'importe quel GPU qui a Vulkan (AMD, NVIDIA ;
//! llvmpipe, le Vulkan de Mesa sur processeur, pour les tests sans GPU).
//!
//! Le flot : les fils de jeu envoient leurs groupes (`gpu::Gpu::submit`) ; un fil « envoi » les
//! recopie dans un emplacement libre de la mémoire que le GPU lit directement (`SLOTS` emplacements,
//! mémoire du GPU visible de l'hôte, sinon mémoire de l'hôte visible du GPU), et soumet le lot : un
//! seul calcul, un groupe de travail par position, tout le réseau dans la mémoire partagée du
//! groupe. Un fil « réception » attend la fin de chaque lot (sémaphore à compteur), lit les sorties
//! là où le GPU les a écrites et répond aux fils de jeu. Pendant que le GPU calcule un lot, le
//! suivant se remplit.
//!
//! Calcul en f32 ; les sommes ne se font pas dans le même ordre que sur le processeur (écarts de
//! l'ordre de 1e-6).
//!
//! `LAB3_VULKAN_PROFILE=1` : la durée de chaque lot sur le GPU (horodatages), totalisée et affichée à
//! la fin.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::mpsc;

use ash::vk;

use crate::encode::{GLOBAL, NODE, REGIONS};
use crate::gpu::{Job, Reply};
use crate::moves::{GLOBAL_KEYS, KEYS};
use crate::net::{Eval, REGION_KINDS};

/// Lots en vol : pendant que le GPU en calcule un, les autres se remplissent ou se lisent. Mesuré le
/// 22 octobre (RX 9070 XT, selfplay, 20 fils) : 943, 1 142, 1 222 et 892 positions/s avec 2, 3, 4 et 6
/// (plus d'emplacements, plus petits lots).
pub const SLOTS: usize = 12;
/// Files de calcul au plus : les lots de files différentes peuvent se chevaucher sur le GPU. Sur
/// RADV, deux files ralentissent (836 positions/s au lieu de 1 222, 4 emplacements).
const QUEUES: u32 = 1;
/// Positions au plus par lot.
pub const MAX_BATCH: usize = 2048;
/// Nombres rendus par position : les logits des coups, puis la tête de valeur.
const OUT: usize = KEYS + 10;
/// La table en tête des poids (`net.comp`, `net32.comp`) : 16 entrées, 8 par couche (8 au plus),
/// puis les poids en bf16 : `node_in` et `node_pol` (parties haute et basse), puis 2 par couche.
const TABLE: usize = 100;
const TABLE16: usize = 80;

const NET16_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/net.spv"));
const NET32_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/net32.spv"));
/// `net.comp` instrumenté (`LAB3_VULKAN_PHASES`) : la durée de chaque étape, par position.
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
const NODE16: usize = 80;

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

/// Où sont les entrées des positions dans un emplacement (octets) : le format de `gpu::Job`.
#[derive(Clone, Copy)]
struct InputLayout {
    node: usize,
    glob: usize,
    adj: usize,
    regions: usize,
    size: usize,
}

impl InputLayout {
    fn new(max: usize) -> InputLayout {
        let up = |x: usize| x.next_multiple_of(16);
        let node = 0;
        let glob = up(node + max * REGIONS * NODE);
        let adj = up(glob + max * GLOBAL);
        let regions = up(adj + max * REGIONS * 8);
        InputLayout { node, glob, adj, regions, size: up(regions + max) }
    }
}

/// Le profil (`LAB3_VULKAN_PROFILE`) : un horodatage au début et à la fin de chaque lot.
struct Profile {
    queries: vk::QueryPool,
    /// Nanosecondes par unité d'horodatage.
    period: f64,
    /// Lots, positions, durée totale sur le GPU (ns).
    totals: Mutex<(u64, u64, f64)>,
}

/// Le réseau sur un GPU, avec `SLOTS` emplacements d'entrées et de sorties.
pub struct Engine {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    /// Les files de calcul ; l'emplacement `s` passe par la file `s % queues.len()`.
    queues: Vec<vk::Queue>,
    command_pool: vk::CommandPool,
    commands: Vec<vk::CommandBuffer>,
    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    sets: Vec<vk::DescriptorSet>,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    module: vk::ShaderModule,
    buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    inputs: Vec<*mut u8>,
    outputs: Vec<*const f32>,
    /// Un sémaphore à compteur par emplacement : il dit la fin de chacun de ses lots.
    timelines: Vec<vk::Semaphore>,
    /// La dernière valeur promise de chaque sémaphore : un seul fil soumet à la fois.
    submitted: Mutex<Vec<u64>>,
    profile: Option<Profile>,
    weights: Weights,
    input_layout: InputLayout,
    max: usize,
    name: String,
    /// Les grands produits en bf16 sur les matrices coopératives (`net.comp`) ; sinon tout en f32
    /// (`net32.comp`).
    bf16: bool,
    /// Le shader instrumenté (`LAB3_VULKAN_PHASES`) : ses durées par étape, 16 par position, dans
    /// `prof` (mémoire projetée).
    phases: bool,
    prof: *mut u32,
}

// SAFETY : les pointeurs sont ceux de la mémoire projetée (`prof` : lu seulement par `bench`), qui vit autant que l'`Engine` ; chaque
// emplacement n'est touché que par le fil qui le tient (voir `serve`), la file et les commandes
// sous `submitted`.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

fn vk_err(what: &str) -> impl Fn(vk::Result) -> String + '_ {
    move |e| format!("Vulkan, {what} : {e}")
}

impl Engine {
    /// Le GPU (le premier GPU dédié, sinon intégré, sinon n'importe quel appareil Vulkan ; ou celui
    /// dont le nom contient `LAB3_VULKAN_DEVICE`) et les poids `net`, pour des lots de `max`
    /// positions au plus.
    pub fn new(net: &Path, max: usize) -> Result<Engine, String> {
        let weights = Weights::load(net)?;
        // SAFETY : appels Vulkan dans l'ordre de la spécification ; chaque objet créé est détruit
        // par `Drop`
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("Vulkan introuvable : {e}"))?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
                .map_err(vk_err("instance"))?;
            let wanted = std::env::var("LAB3_VULKAN_DEVICE").ok();
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
            let bf16 = std::env::var_os("LAB3_VULKAN_F32").is_none() && Engine::has_bf16(&entry, &instance, pd);
            let phases = bf16 && std::env::var_os("LAB3_VULKAN_PHASES").is_some();
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
                descriptor_pool: vk::DescriptorPool::null(),
                set_layout: vk::DescriptorSetLayout::null(),
                sets: Vec::new(),
                layout: vk::PipelineLayout::null(),
                pipeline: vk::Pipeline::null(),
                module: vk::ShaderModule::null(),
                buffers: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                timelines: Vec::new(),
                submitted: Mutex::new(vec![0; SLOTS]),
                profile: None,
                input_layout: InputLayout::new(max),
                weights,
                max,
                name,
                bf16,
                phases,
                prof: std::ptr::null_mut(),
            };
            engine.build(&memory, family)?;
            if std::env::var_os("LAB3_VULKAN_PROFILE").is_some() {
                let queries = engine
                    .device
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2 * SLOTS as u32),
                        None,
                    )
                    .map_err(vk_err("horodatages"))?;
                engine.profile = Some(Profile { queries, period: period as f64, totals: Mutex::new((0, 0, 0.0)) });
            }
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
            // entrées : écrites par l'hôte, lues par le GPU (de préférence dans sa mémoire, si
            // l'hôte la voit) ; sorties : écrites par le GPU, lues par l'hôte (de préférence en
            // cache de l'hôte)
            let mut slots = Vec::new();
            for _ in 0..SLOTS {
                let (ibuf, imem) =
                    self.buffer(memory, self.input_layout.size, storage, &[M::DEVICE_LOCAL | host, host])?;
                let (obuf, omem) = self.buffer(memory, self.max * OUT * 4, storage, &[host | M::HOST_CACHED, host])?;
                let map = |m| self.device.map_memory(m, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty());
                let ip = map(imem).map_err(vk_err("projection"))?;
                let op = map(omem).map_err(vk_err("projection"))?;
                self.inputs.push(ip as *mut u8);
                self.outputs.push(op as *const f32);
                slots.push((ibuf, obuf));
            }

            // les durées par étape (`phases`) : toujours là, pour une seule disposition
            let (pbuf, pmem) = self.buffer(memory, self.max * 16 * 4, storage, &[host])?;
            let pp = self
                .device
                .map_memory(pmem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(vk_err("projection"))?;
            self.prof = pp as *mut u32;

            let d = &self.device;
            let bindings: Vec<_> = (0..5)
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
            let push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(20)];
            let set_layouts = [self.set_layout];
            self.layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&push),
                    None,
                )
                .map_err(vk_err("disposition"))?;
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(5 * SLOTS as u32)];
            self.descriptor_pool = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(SLOTS as u32).pool_sizes(&sizes),
                    None,
                )
                .map_err(vk_err("réserve"))?;
            let layouts = vec![self.set_layout; SLOTS];
            self.sets = d
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&layouts),
                )
                .map_err(vk_err("descripteurs"))?;
            for (&set, &(ibuf, obuf)) in self.sets.iter().zip(&slots) {
                let infos: Vec<[vk::DescriptorBufferInfo; 1]> = [wbuf, ibuf, obuf, w16buf, pbuf]
                    .iter()
                    .map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)])
                    .collect();
                let writes: Vec<_> = infos
                    .iter()
                    .enumerate()
                    .map(|(i, info)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(i as u32)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(info)
                    })
                    .collect();
                d.update_descriptor_sets(&writes, &[]);
            }

            let code = match (self.bf16, self.phases) {
                (true, true) => NET16_PROF_SPV,
                (true, false) => NET16_SPV,
                _ => NET32_SPV,
            };
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
                        .command_buffer_count(SLOTS as u32),
                )
                .map_err(vk_err("commandes"))?;
            for _ in 0..SLOTS {
                let mut kind = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
                let sem = d
                    .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut kind), None)
                    .map_err(vk_err("sémaphore"))?;
                self.timelines.push(sem);
            }

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

    /// La mémoire des entrées de l'emplacement `slot`.
    #[allow(clippy::mut_from_ref)]
    fn input(&self, slot: usize) -> &mut [u8] {
        // SAFETY : mémoire projetée de `input_layout.size` octets ; l'emplacement appartient à
        // l'appelant (`serve`)
        unsafe { std::slice::from_raw_parts_mut(self.inputs[slot], self.input_layout.size) }
    }

    /// Les sorties de l'emplacement `slot` : `OUT` nombres par position.
    fn output(&self, slot: usize) -> &[f32] {
        // SAFETY : mémoire projetée de `max · OUT` nombres
        unsafe { std::slice::from_raw_parts(self.outputs[slot], self.max * OUT) }
    }

    /// Range les positions du groupe `job` à partir de la position `at` de l'emplacement `slot`.
    fn write(&self, slot: usize, at: usize, job: &Job) {
        let (l, input) = (self.input_layout, self.input(slot));
        let mut put = |off: usize, size: usize, data: &[u8]| {
            input[off + at * size..off + (at + job.count) * size].copy_from_slice(data);
        };
        put(l.node, REGIONS * NODE, &job.node);
        put(l.glob, GLOBAL, &job.glob);
        put(l.adj, REGIONS * 8, &job.adj);
        put(l.regions, 1, &job.regions);
    }

    /// Soumet le calcul des `count` positions de l'emplacement `slot` ; rend la valeur du
    /// sémaphore qui dira sa fin (`wait`).
    fn submit(&self, slot: usize, count: usize) -> Result<u64, String> {
        assert!(count <= self.max, "lot trop grand");
        let mut submitted = self.submitted.lock().expect("verrou");
        let cb = self.commands[slot];
        let l = self.input_layout;
        let params: Vec<u8> = [l.node, l.glob, l.adj, l.regions, self.weights.layers as usize]
            .iter()
            .flat_map(|&v| (v as u32).to_le_bytes())
            .collect();
        // SAFETY : la commande de l'emplacement n'est plus en vol (l'appelant a attendu sa fin) ;
        // un seul fil soumet à la fois (`submitted`)
        unsafe {
            let d = &self.device;
            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()).map_err(vk_err("commandes"))?;
            d.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(vk_err("commandes"))?;
            if let Some(p) = &self.profile {
                d.cmd_reset_query_pool(cb, p.queries, 2 * slot as u32, 2);
                d.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, p.queries, 2 * slot as u32);
            }
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[self.sets[slot]], &[]);
            d.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &params);
            d.cmd_dispatch(cb, count as u32, 1, 1);
            if let Some(p) = &self.profile {
                d.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, p.queries, 2 * slot as u32 + 1);
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
            submitted[slot] += 1;
            let values = [submitted[slot]];
            let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
            let (cbs, sems) = ([cb], [self.timelines[slot]]);
            let info =
                vk::SubmitInfo::default().command_buffers(&cbs).signal_semaphores(&sems).push_next(&mut timeline);
            let queue = self.queues[slot % self.queues.len()];
            d.queue_submit(queue, &[info], vk::Fence::null()).map_err(vk_err("envoi"))?;
        }
        Ok(submitted[slot])
    }

    /// Attend la fin du lot de l'emplacement `slot` soumis avec la valeur `value`.
    fn wait(&self, slot: usize, value: u64) -> Result<(), String> {
        let (sems, values) = ([self.timelines[slot]], [value]);
        // SAFETY : sémaphore vivant ; l'attente peut se faire de n'importe quel fil
        unsafe {
            self.device.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&values), u64::MAX)
        }
        .map_err(vk_err("attente"))
    }

    /// Durée moyenne sur le GPU (ms) d'un lot de `count` positions de `regions` régions (en anneau,
    /// deux voisines chacune), sur `rounds` lots ; demande `LAB3_VULKAN_PROFILE`.
    pub fn bench(&self, count: usize, regions: usize, rounds: usize) -> Result<f64, String> {
        let l = self.input_layout;
        let input = self.input(0);
        input.fill(0);
        for pos in 0..count {
            input[l.regions + pos] = regions as u8;
            for i in 0..regions {
                let bits = (1u64 << ((i + 1) % regions)) | (1u64 << ((i + regions - 1) % regions));
                let at = l.adj + (pos * REGIONS + i) * 8;
                input[at..at + 8].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let p = self.profile.as_ref().ok_or("LAB3_VULKAN_PROFILE=1 attendu")?;
        *p.totals.lock().expect("verrou") = (0, 0, 0.0);
        for _ in 0..rounds {
            let v = self.submit(0, count)?;
            self.wait(0, v)?;
            self.profile_lot(0, count);
        }
        let (lots, _, ns) = *p.totals.lock().expect("verrou");
        Ok(ns / 1e6 / lots.max(1) as f64)
    }

    /// En flux : `lots` lots de `count` positions soumis à la suite, `SLOTS` en vol au plus, comme
    /// le selfplay ; rend la durée murale totale (ms) et la somme des durées sur le GPU (ms). Leur
    /// écart : ce que coûtent les soumissions et les fins de lot.
    pub fn bench_stream(&self, count: usize, regions: usize, lots: usize) -> Result<(f64, f64), String> {
        // les mêmes positions dans tous les emplacements
        self.bench(count, regions, 1)?;
        let first = self.input(0).to_vec();
        for slot in 1..SLOTS {
            self.input(slot).copy_from_slice(&first);
        }
        let p = self.profile.as_ref().ok_or("LAB3_VULKAN_PROFILE=1 attendu")?;
        *p.totals.lock().expect("verrou") = (0, 0, 0.0);
        let start = std::time::Instant::now();
        let mut pending = std::collections::VecDeque::new();
        for k in 0..lots {
            let slot = k % SLOTS;
            if pending.len() == SLOTS {
                let (s, v) = pending.pop_front().expect("un lot");
                self.wait(s, v)?;
                self.profile_lot(s, count);
            }
            pending.push_back((slot, self.submit(slot, count)?));
        }
        for (s, v) in pending {
            self.wait(s, v)?;
            self.profile_lot(s, count);
        }
        let wall = start.elapsed().as_secs_f64() * 1e3;
        let (_, _, ns) = *p.totals.lock().expect("verrou");
        Ok((wall, ns / 1e6))
    }

    /// Avec `LAB3_VULKAN_PHASES` : la durée moyenne de chaque étape (`PHASES`, µs, horloge du GPU à
    /// 100 MHz) d'une position, sur les `rounds` lots de `count` positions de `bench`.
    pub fn bench_phases(&self, count: usize, regions: usize, rounds: usize) -> Result<Option<Vec<f64>>, String> {
        if !self.phases {
            return Ok(None);
        }
        // SAFETY : mémoire projetée de `max · 16` nombres, que le GPU ne touche qu'entre `submit`
        // et `wait`
        let prof = unsafe { std::slice::from_raw_parts_mut(self.prof, self.max * 16) };
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

    /// Ajoute au profil la durée du lot fini de l'emplacement `slot` (`count` positions).
    fn profile_lot(&self, slot: usize, count: usize) {
        let Some(p) = &self.profile else { return };
        let mut stamps = [0u64; 2];
        // SAFETY : le lot est fini (`wait`) ; horodatages écrits par ses commandes
        let read = unsafe {
            self.device.get_query_pool_results(
                p.queries,
                2 * slot as u32,
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
            for &sem in &self.timelines {
                d.destroy_semaphore(sem, None);
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

/// Un lot reçu : son emplacement revient à l'envoi quand le dernier groupe a lu ses sorties.
struct Lot {
    free: mpsc::Sender<usize>,
    slot: usize,
}

impl Drop for Lot {
    fn drop(&mut self) {
        let _ = self.free.send(self.slot);
    }
}

/// Les sorties d'un emplacement, lues par les fils de jeu.
#[derive(Clone, Copy)]
struct Outputs(*const f32);
// SAFETY : mémoire projetée qui vit autant que l'`Engine`, lue seulement (voir `Lot`)
unsafe impl Send for Outputs {}

/// L'évaluation d'une position : `OUT` nombres rendus par le GPU.
fn eval_of(players: usize, out: &[f32]) -> Eval {
    let v: &[f32; 10] = out[KEYS..OUT].try_into().expect("10 valeurs");
    Eval::from_heads(players, out[..KEYS].to_vec(), v)
}

/// Le service des fils de jeu (`gpu::Gpu::start_vulkan`) : envoi et réception des lots, chacun
/// sur son fil (voir l'en-tête).
pub(crate) fn serve(net: &Path, jobs: mpsc::Receiver<Job>, ready: mpsc::Sender<Result<String, String>>) {
    let engine = match Engine::new(net, MAX_BATCH) {
        Ok(e) => e,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let _ = ready.send(Ok(engine.describe()));
    let engine = &engine;
    let (free, slots) = mpsc::channel::<usize>();
    for s in 0..SLOTS {
        free.send(s).expect("emplacements");
    }
    let (sent, received) = mpsc::channel::<(u64, usize, Vec<Job>)>();
    let profile = engine.profile.is_some();
    std::thread::scope(|scope| {
        // réception : attend chaque lot, dans l'ordre, et rend son emplacement
        let thread = std::thread::Builder::new().name("vulkan-réception".into());
        thread
            .spawn_scoped(scope, move || {
                let mut busy = std::time::Duration::ZERO;
                let start = std::time::Instant::now();
                for (value, slot, batch) in received {
                    engine.wait(slot, value).unwrap_or_else(|e| panic!("{e}"));
                    let began = std::time::Instant::now();
                    engine.profile_lot(slot, batch.iter().map(|j| j.count).sum());
                    // chaque fil de jeu lit ses sorties lui-même ; l'emplacement revient quand tous
                    // les groupes du lot les ont lues (ou abandonnées)
                    let lot = std::sync::Arc::new(Lot { free: free.clone(), slot });
                    let out = Outputs(engine.output(slot).as_ptr());
                    let mut at = 0;
                    for job in batch {
                        let (lot, players, count) = (lot.clone(), job.players, job.count);
                        let reply: Reply = Box::new(move || {
                            // toute la structure (pas son seul champ, qui n'est pas `Send`)
                            let (_lot, out) = (lot, out);
                            // SAFETY : les sorties de ce lot, écrites par le GPU (lot fini) ;
                            // l'emplacement n'est réutilisé qu'après la fin de `_lot`
                            let out = unsafe { std::slice::from_raw_parts(out.0.add(at * OUT), count * OUT) };
                            (0..count).map(|k| eval_of(players[k] as usize, &out[k * OUT..])).collect()
                        });
                        at += count;
                        // un fil de jeu arrêté ne relit pas : rien à faire
                        let _ = job.reply.send(reply);
                    }
                    busy += began.elapsed();
                }
                if profile {
                    eprintln!("réception occupée {:.0} %", 100.0 * busy.as_secs_f64() / start.elapsed().as_secs_f64());
                }
            })
            .expect("fil de réception");
        // envoi : un emplacement libre, puis tous les groupes qui attendent (au moins un)
        let mut carry: Option<Job> = None;
        let (mut busy, mut idle_slot, mut idle_jobs) =
            (std::time::Duration::ZERO, std::time::Duration::ZERO, std::time::Duration::ZERO);
        let start = std::time::Instant::now();
        loop {
            let waited = std::time::Instant::now();
            let Ok(slot) = slots.recv() else { break };
            idle_slot += waited.elapsed();
            let waited = std::time::Instant::now();
            let first = match carry.take() {
                Some(job) => job,
                None => match jobs.recv() {
                    Ok(job) => job,
                    Err(_) => break,
                },
            };
            idle_jobs += waited.elapsed();
            let began = std::time::Instant::now();
            let mut count = first.count;
            let mut batch = vec![first];
            for job in jobs.try_iter() {
                if count + job.count > MAX_BATCH {
                    carry = Some(job);
                    break;
                }
                count += job.count;
                batch.push(job);
            }
            let mut at = 0;
            for job in &batch {
                engine.write(slot, at, job);
                at += job.count;
            }
            let value = engine.submit(slot, count).unwrap_or_else(|e| panic!("{e}"));
            busy += began.elapsed();
            if sent.send((value, slot, batch)).is_err() {
                break;
            }
        }
        if profile {
            let all = start.elapsed().as_secs_f64() / 100.0;
            eprintln!(
                "envoi : occupé {:.0} %, attend un emplacement {:.0} %, attend des groupes {:.0} %",
                busy.as_secs_f64() / all,
                idle_slot.as_secs_f64() / all,
                idle_jobs.as_secs_f64() / all
            );
        }
        drop(sent);
    });
}
