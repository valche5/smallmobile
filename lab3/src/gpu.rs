//! Le réseau sur GPU par Vulkan, pour la génération des parties (`lab3 selfplay --device vulkan`).
//! Les fils de jeu envoient leurs groupes de positions (`batch::Batch`) au service Vulkan, qui
//! évalue ensemble les groupes qui attendent. Chaque fil de jeu mène plusieurs groupes
//! (`selfplay::run`) : pendant que le GPU en évalue, il fait avancer les autres.
//!
//! Les nombres du GPU (bf16, sommes dans un autre ordre) diffèrent un peu de ceux du processeur :
//! les parties ne sont plus les mêmes au bit près que sans GPU.

use std::path::Path;
use std::sync::mpsc;

use crate::encode::{GLOBAL, NODE, REGIONS};
use crate::net::{Eval, Position};

/// Un groupe envoyé au service Vulkan.
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
pub(crate) struct Job {
    pub(crate) count: usize,
    pub(crate) node: Vec<u8>,
    pub(crate) glob: Vec<u8>,
    pub(crate) adj: Vec<u8>,
    pub(crate) regions: Vec<u8>,
    pub(crate) players: Vec<u8>,
    pub(crate) reply: mpsc::Sender<Reply>,
}

/// La réponse à un groupe : ses évaluations, à tirer par le fil de jeu qui les attend (Vulkan : il
/// les lit lui-même dans la mémoire où le GPU les a écrites, `vulkan::serve`).
pub(crate) type Reply = Box<dyn FnOnce() -> Vec<Eval> + Send>;

/// Les évaluations d'un groupe envoyé (`Gpu::submit`).
pub struct Pending(mpsc::Receiver<Reply>);

impl Pending {
    pub fn wait(self) -> Vec<Eval> {
        self.receive()()
    }

    /// La réponse seule, sans la décoder (pour mesurer à part l'attente et le décodage).
    pub(crate) fn receive(self) -> Reply {
        self.0.recv().expect("le fil du GPU s'est arrêté")
    }
}

pub struct Gpu {
    jobs: Option<mpsc::Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Gpu {
    /// Lance le service Vulkan avec les poids `net` ; rend aussi l'appareil qui calcule.
    pub fn start(net: &Path) -> Result<(Gpu, String), String> {
        #[cfg(feature = "vulkan")]
        return Gpu::launch(net);
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = net;
            Err("programme lab3 compilé sans Vulkan (cargo build --features vulkan)".into())
        }
    }

    #[cfg(feature = "vulkan")]
    fn launch(net: &Path) -> Result<(Gpu, String), String> {
        let (jobs, received) = mpsc::channel();
        let (ready, started) = mpsc::channel();
        let net = net.to_owned();
        // L'envoi des lots à Vulkan, nommé pour les profils.
        let thread = std::thread::Builder::new()
            .name("gpu-envoi".into())
            .spawn(move || crate::vulkan::serve(&net, received, ready))
            .map_err(|e| format!("fil du GPU : {e}"))?;
        match started.recv() {
            Ok(Ok(device)) => Ok((Gpu { jobs: Some(jobs), thread: Some(thread) }, device)),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("le fil du GPU s'est arrêté au démarrage".into()),
        }
    }

    /// Envoie un groupe de positions ; ses évaluations viendront par `Pending::wait`, dans l'ordre.
    pub fn submit<P: Position>(&self, positions: &[P], players: impl Iterator<Item = usize>) -> Pending {
        let count = positions.len();
        let mut job = Job {
            count,
            node: Vec::with_capacity(count * REGIONS * NODE),
            glob: Vec::with_capacity(count * GLOBAL),
            adj: Vec::with_capacity(count * REGIONS * 8),
            regions: Vec::with_capacity(count),
            players: players.map(|p| p as u8).collect(),
            reply: mpsc::channel().0,
        };
        for p in positions {
            let e = p.encoded();
            job.node.extend_from_slice(e.node.as_flattened());
            job.glob.extend_from_slice(&e.global);
            for i in 0..REGIONS {
                let bits = if i < e.regions { p.adj(i).iter().fold(0u64, |b, &j| b | 1 << j) } else { 0 };
                job.adj.extend_from_slice(&bits.to_le_bytes());
            }
            job.regions.push(e.regions as u8);
        }
        let (reply, pending) = mpsc::channel();
        job.reply = reply;
        self.jobs.as_ref().expect("GPU arrêté").send(job).expect("le fil du GPU s'est arrêté");
        Pending(pending)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
