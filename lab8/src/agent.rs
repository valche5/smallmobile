//! Le réseau de lab8 comme joueur des arènes de lab7 : le décodeur seul, sans recherche
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
use crate::plan::{Writer, pick};
use crate::rand::Rng;
use crate::search;
use crate::tokens::distances;

pub struct Plan {
    pub net: Arc<Net>,
    pub label: String,
    pub t: f32,
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
        Some(Plan { net: Arc::new(loaded), label: path, t })
    }
}

impl Agent for Plan {
    fn name(&self) -> String {
        format!("plan({}, t={})", self.label, self.t)
    }

    /// Une décision du jeu (un coup, ou toute une fin de tour, une défense), écrite par le décodeur
    /// depuis une lecture de la position au début de la décision.
    fn act(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Vec<Action> {
        let dist = distances(g.map);
        let mut writer: Option<Writer> = None;
        let choose = |w: &Game, kw: &Knowledge| {
            let legal = moves::legal(w, kw);
            if legal.len() == 1 {
                return legal[0].mv;
            }
            let wr = writer.get_or_insert_with(|| Writer::new(&self.net, w, kw, &dist));
            let logits = wr.logits(w, kw, &legal);
            let c = legal[pick(&logits, self.t, rng)];
            wr.played(c.key);
            c.mv
        };
        moves::decision(g, k, choose).expect("coup du décodeur").0
    }
}

/// La recherche sur des plans (`search`) comme joueur des arènes : `plans:net=…[,k=16,m=4,t=1,mix=0.5,
/// win=0.5,head=final]`. Le plan choisi couvre souvent plusieurs décisions du jeu (les conquêtes une à
/// une) : sa suite est gardée, sous l'empreinte de la position où elle reprend.
pub struct Plans {
    pub net: Arc<Net>,
    pub label: String,
    pub cfg: search::Config,
    pending: Mutex<HashMap<u64, Vec<Move>>>,
    /// L'encodeur sur GPU (`gpu=1`), partagé par les fils de l'arène.
    #[cfg(feature = "vulkan")]
    gpu: Option<Arc<crate::vulkan::Shared>>,
}

#[cfg(feature = "vulkan")]
impl search::Reader for crate::vulkan::Shared {
    fn read(&self, items: Vec<(crate::encode::Encoded, Vec<u8>)>) -> Vec<crate::net::Memory> {
        self.encode(items)
    }
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
        let (mut net, mut rest, mut gpu) = (None, Vec::new(), false);
        for part in opts.split(',') {
            match part.split_once('=')? {
                ("net", path) => net = Some(path.to_string()),
                ("gpu", v) => gpu = v == "1",
                _ => rest.push(part),
            }
        }
        let cfg = search::Config::parse(&rest.join(","))?;
        let path = net?;
        let loaded = Net::load(Path::new(&path)).unwrap_or_else(|e| panic!("{path} : {e}"));
        #[cfg(feature = "vulkan")]
        let gpu = gpu.then(|| {
            let encoder = crate::vulkan::Encoder::new(Path::new(&path), 1024).unwrap_or_else(|e| panic!("GPU : {e}"));
            eprintln!("encodeur de lab8 : {}", encoder.describe());
            Arc::new(crate::vulkan::Shared::new(encoder))
        });
        #[cfg(not(feature = "vulkan"))]
        assert!(!gpu, "gpu=1 : compilé sans Vulkan");
        let label = format!("{path}{}{}", if rest.is_empty() { "" } else { ", " }, rest.join(","));
        Some(Plans {
            net: Arc::new(loaded),
            label,
            cfg,
            pending: Mutex::default(),
            #[cfg(feature = "vulkan")]
            gpu,
        })
    }
}

impl Agent for Plans {
    fn name(&self) -> String {
        format!("plans({})", self.label)
    }

    fn act(&self, g: &Game, k: &Knowledge, rng: &mut Rng) -> Vec<Action> {
        let cached = self.pending.lock().expect("plans gardés").remove(&position_key(g, k));
        if cached.is_some() {
            RESUMED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        #[cfg(feature = "vulkan")]
        let reader: &dyn search::Reader = match &self.gpu {
            Some(gpu) => gpu.as_ref(),
            None => self.net.as_ref(),
        };
        #[cfg(not(feature = "vulkan"))]
        let reader: &dyn search::Reader = self.net.as_ref();
        let mut plan =
            cached.unwrap_or_else(|| search::search(&self.net, reader, &self.cfg, g, k, rng).plan).into_iter();
        let (actions, played) = moves::decision(g, k, |w, kw| {
            plan.next().unwrap_or_else(|| {
                // le plan est fini avant la décision du jeu : le plus probable du décodeur
                SHORT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                crate::plan::write(&self.net, w, kw, &distances(w.map), 0.0, &mut Rng::new(0)).0[0]
            })
        })
        .expect("coup de la recherche");
        let rest: Vec<Move> = plan.collect();
        if !rest.is_empty() {
            let (mut w, mut kw) = (*g, *k);
            for mv in &played {
                kw.play(&mut w, mv).expect("coup joué");
            }
            self.pending.lock().expect("plans gardés").insert(position_key(&w, &kw), rest);
        }
        actions
    }
}
