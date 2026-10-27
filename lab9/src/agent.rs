//! Le réseau de lab9 comme joueur des arènes de lab7 : le décodeur seul, sans recherche
//! (`plan:net=…[,t=…]`, température 0 par défaut : le coup le plus probable), ou la recherche sur des
//! plans (`plans:net=…,k=…`).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex};

use smallmobile_kernel::{Action, Game};
use smallmobile_lab7::agent::Agent;

use crate::know::Knowledge;
use crate::moves::{self, Move};
use crate::net::Net;
use crate::rand::Rng;
use crate::search;
use crate::tokens::distances;

pub struct Plan {
    pub net: Arc<Net>,
    pub label: String,
    pub t: f32,
    pending: Pending,
}

/// Les suites de plans gardées entre deux décisions du jeu : un plan couvre souvent plusieurs décisions
/// du jeu (les conquêtes une à une) ; le décodeur l'écrit d'un trait depuis son début, comme à
/// l'apprentissage, et la suite reprend sous l'empreinte de la position où elle commence.
#[derive(Default)]
struct Pending(Mutex<HashMap<u64, Vec<Move>>>);

impl Pending {
    /// La décision du jeu en `g` : la suite gardée pour cette position, sinon le plan `fresh`, jouée
    /// jusqu'au bout de la décision ; le reste est gardé.
    fn act(&self, net: &Net, g: &Game, k: &Knowledge, fresh: impl FnOnce() -> Vec<Move>) -> Vec<Action> {
        let cached = self.0.lock().expect("plans gardés").remove(&position_key(g, k));
        if cached.is_some() {
            RESUMED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let mut plan = cached.unwrap_or_else(fresh).into_iter();
        let (actions, played) = moves::decision(g, k, |w, kw| {
            plan.next().unwrap_or_else(|| {
                // le plan est fini avant la décision du jeu : le plus probable du décodeur
                SHORT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::plan::write(net, w, kw, &distances(w.map), 0.0, &mut Rng::new(0)).0[0]
            })
        })
        .expect("coup du plan");
        if std::env::var_os("LAB9_TRACE").is_some() {
            let (mut w, mut kw) = (*g, *k);
            let mut line = format!("manche {} joueur {} :", g.state.round, g.state.actor().unwrap_or(9));
            for mv in &played {
                line += &format!(" {}", moves::describe(&w, mv));
                kw.play(&mut w, mv).expect("coup joué");
            }
            eprintln!("{line}");
        }
        let rest: Vec<Move> = plan.collect();
        if !rest.is_empty() {
            let (mut w, mut kw) = (*g, *k);
            for mv in &played {
                kw.play(&mut w, mv).expect("coup joué");
            }
            self.0.lock().expect("plans gardés").insert(position_key(&w, &kw), rest);
        }
        actions
    }
}

impl Plan {
    /// `plan:net=chemin[,t=0.5]`.
    pub fn named(name: &str) -> Option<Plan> {
        let opts = name.strip_prefix("plan:")?;
        let (mut net, mut t) = (None, 0.0);
        for part in opts.split(',') {
            match part.split_once('=')? {
                ("net", path) => net = Some(path.to_string()),
                ("t", v) => t = v.parse().ok()?,
                _ => return None,
            }
        }
        let path = net?;
        let loaded = Net::load(Path::new(&path)).unwrap_or_else(|e| panic!("{path} : {e}"));
        Some(Plan { net: Arc::new(loaded), label: path, t, pending: Pending::default() })
    }
}

impl Agent for Plan {
    fn name(&self) -> String {
        format!("plan({}, t={})", self.label, self.t)
    }

    /// Une décision du jeu (un coup, ou toute une fin de tour, une défense) : la suite du plan en cours,
    /// ou un plan neuf écrit d'un trait par le décodeur.
    fn act(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Vec<Action> {
        self.pending.act(&self.net, g, k, || crate::plan::write(&self.net, g, k, &distances(g.map), self.t, rng).0)
    }
}

/// La recherche sur des plans (`search`) comme joueur des arènes : `plans:net=…[,k=16,m=4,t=1,mix=0.5,
/// win=0.5,head=final]`. Le plan choisi couvre souvent plusieurs décisions du jeu (les conquêtes une à
/// une) : sa suite est gardée, sous l'empreinte de la position où elle reprend.
pub struct Plans {
    pub net: Arc<Net>,
    pub label: String,
    pub cfg: search::Config,
    pending: Pending,
    /// Le moteur GPU (`gpu=1` : encodeur et décodeur ; `gpu=enc` : l'encodeur seul, le décodeur sur
    /// le processeur), partagé par les fils de l'arène.
    #[cfg(feature = "vulkan")]
    gpu: Option<(crate::vulkan::Gpu, bool)>,
}

/// Décisions du jeu où le plan choisi était fini trop tôt (jamais attendu : compté pour les tests), et
/// suites de plans reprises.
pub static SHORT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static RESUMED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// L'empreinte de ce que voit le joueur qui agit.
fn position_key(g: &Game, k: &Knowledge) -> u64 {
    let e = crate::encode::encode(g, k);
    let mut h = std::hash::DefaultHasher::new();
    (e.regions, g.state.actor(), g.map.seed).hash(&mut h);
    e.node[..e.regions].hash(&mut h);
    e.global.hash(&mut h);
    h.finish()
}

impl Plans {
    pub fn named(name: &str) -> Option<Plans> {
        let opts = name.strip_prefix("plans:")?;
        let (mut net, mut rest, mut gpu, mut decoder) = (None, Vec::new(), false, true);
        for part in opts.split(',') {
            match part.split_once('=')? {
                ("net", path) => net = Some(path.to_string()),
                ("gpu", "1") => gpu = true,
                ("gpu", "enc") => (gpu, decoder) = (true, false),
                ("gpu", _) => return None,
                _ => rest.push(part),
            }
        }
        let cfg = search::Config::parse(&rest.join(","))?;
        if gpu {
            rest.push(if decoder { "gpu=1" } else { "gpu=enc" });
        }
        let path = net?;
        let loaded = Net::load(Path::new(&path)).unwrap_or_else(|e| panic!("{path} : {e}"));
        #[cfg(feature = "vulkan")]
        let gpu = gpu.then(|| {
            let encoder = crate::vulkan::Encoder::new(Path::new(&path), 1024).unwrap_or_else(|e| panic!("GPU : {e}"));
            eprintln!("encodeur de lab9 : {}", encoder.describe());
            (crate::vulkan::Gpu(Arc::new(crate::vulkan::Shared::new(encoder))), decoder)
        });
        #[cfg(not(feature = "vulkan"))]
        let _ = decoder;
        #[cfg(not(feature = "vulkan"))]
        assert!(!gpu, "gpu=1 : compilé sans Vulkan");
        let label = format!("{path}{}{}", if rest.is_empty() { "" } else { ", " }, rest.join(","));
        Some(Plans {
            net: Arc::new(loaded),
            label,
            cfg,
            pending: Pending::default(),
            #[cfg(feature = "vulkan")]
            gpu,
        })
    }
}

/// `LAB9_GPU_PROFILE` : le temps du GPU par kernel, à la fin de l'arène.
#[cfg(feature = "vulkan")]
impl Drop for Plans {
    fn drop(&mut self) {
        if let Some(p) = self.gpu.as_ref().and_then(|(gpu, _)| gpu.0.encoder.profile()) {
            eprintln!("GPU de {} : {p}", self.label);
        }
    }
}

impl Agent for Plans {
    fn name(&self) -> String {
        format!("plans({})", self.label)
    }

    fn act(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Vec<Action> {
        self.pending.act(&self.net, g, k, || {
            #[cfg(feature = "vulkan")]
            match &self.gpu {
                Some((gpu, true)) => return search::search(gpu, &self.cfg, g, k, rng).plan,
                Some((gpu, false)) => {
                    let engine = crate::vulkan::GpuEncoder(&gpu.0, &self.net);
                    return search::search(&engine, &self.cfg, g, k, rng).plan;
                }
                None => {}
            }
            search::search(&search::Cpu(&self.net), &self.cfg, g, k, rng).plan
        })
    }
}
