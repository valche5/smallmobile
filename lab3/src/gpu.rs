//! Le réseau sur GPU, pour la génération des parties (`lab3 selfplay --gpu`, programme compilé avec
//! `--features gpu`) : un fil « évaluateur » appelle PyTorch (`lab3/py/serve.py`, par pyo3) ; les
//! fils de jeu lui envoient chacun leur groupe de positions (`batch::Batch`), et il évalue ensemble
//! tous les groupes qui attendent. Chaque fil de jeu mène plusieurs groupes (`selfplay::run`) :
//! pendant que le GPU en évalue, il fait avancer les autres. Seul le fil évaluateur touche à Python.
//!
//! Les nombres du GPU (bf16, sommes dans un autre ordre) diffèrent un peu de ceux du processeur :
//! les parties ne sont plus les mêmes au bit près que sans GPU.
//!
//! Python est celui de `lab3/py/.venv` (PyTorch pour CUDA ou ROCm : `lab3/py/setup.sh`), là où le
//! programme a été compilé.
//!
//! Le même service sans Python, par Vulkan (`Gpu::start_vulkan`, programme compilé avec
//! `--features vulkan`) : `vulkan`.

use std::path::Path;
use std::sync::mpsc;

use crate::encode::{GLOBAL, NODE, REGIONS};
use crate::net::{Eval, Position};

/// Un groupe envoyé au GPU, dans le format de `serve.py` (et de `vulkan::serve`).
#[cfg_attr(not(any(feature = "gpu", feature = "vulkan")), allow(dead_code))]
pub(crate) struct Job {
    pub(crate) count: usize,
    pub(crate) node: Vec<u8>,
    pub(crate) glob: Vec<u8>,
    pub(crate) adj: Vec<u8>,
    pub(crate) regions: Vec<u8>,
    pub(crate) players: Vec<u8>,
    pub(crate) reply: mpsc::Sender<Vec<Eval>>,
}

/// Un service des groupes : poids, groupes reçus, et ce qui calcule (ou pourquoi rien) dès le départ.
type Serve = fn(&Path, mpsc::Receiver<Job>, mpsc::Sender<Result<String, String>>);

/// Les évaluations d'un groupe envoyé (`Gpu::submit`).
pub struct Pending(mpsc::Receiver<Vec<Eval>>);

impl Pending {
    pub fn wait(self) -> Vec<Eval> {
        self.0.recv().expect("le fil du GPU s'est arrêté")
    }
}

pub struct Gpu {
    jobs: Option<mpsc::Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Gpu {
    /// Lance le fil évaluateur avec les poids `net` ; rend aussi ce qui calcule (GPU ou processeur).
    pub fn start(net: &Path) -> Result<(Gpu, String), String> {
        Gpu::launch(net, serve)
    }

    /// Le même service par Vulkan, sans Python (`vulkan`).
    pub fn start_vulkan(net: &Path) -> Result<(Gpu, String), String> {
        #[cfg(feature = "vulkan")]
        return Gpu::launch(net, crate::vulkan::serve);
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = net;
            Err("programme lab3 compilé sans Vulkan (cargo build --features vulkan)".into())
        }
    }

    fn launch(net: &Path, serve: Serve) -> Result<(Gpu, String), String> {
        let (jobs, received) = mpsc::channel();
        let (ready, started) = mpsc::channel();
        let net = net.to_owned();
        let thread = std::thread::spawn(move || serve(&net, received, ready));
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

/// Le fil évaluateur : les groupes qui attendent, évalués ensemble, jusqu'à la fin des envois.
#[cfg(feature = "gpu")]
fn serve(net: &Path, jobs: mpsc::Receiver<Job>, ready: mpsc::Sender<Result<String, String>>) {
    use pyo3::prelude::*;
    use pyo3::types::{PyByteArray, PyBytes};

    use crate::moves::KEYS;

    pyo3::Python::attach(|py| {
        let start = || -> PyResult<(Bound<PyAny>, String)> {
            // `lab3/py` et son environnement, là où le programme a été compilé
            let py_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/py");
            let sys = py.import("sys")?;
            sys.getattr("path")?.call_method1("insert", (0, py_dir))?;
            let site = py.import("site")?;
            for d in py
                .import("glob")?
                .call_method1("glob", (format!("{py_dir}/.venv/lib/python3*/site-packages"),))?
                .try_iter()?
            {
                site.call_method1("addsitedir", (d?,))?;
            }
            let evaluator = py.import("serve")?.getattr("Evaluator")?.call1((net.to_string_lossy().into_owned(),))?;
            let device: String = evaluator.call_method0("describe")?.extract()?;
            Ok((evaluator, device))
        };
        let evaluator = match start() {
            Ok((evaluator, device)) => {
                let _ = ready.send(Ok(device));
                evaluator
            }
            Err(e) => {
                let _ = ready.send(Err(format!("PyTorch : {e}")));
                return;
            }
        };
        while let Ok(first) = jobs.recv() {
            // tous les groupes qui attendent déjà
            let mut batch = vec![first];
            batch.extend(jobs.try_iter());
            let count: usize = batch.iter().map(|j| j.count).sum();
            let join = |part: fn(&Job) -> &Vec<u8>| {
                PyByteArray::new(py, &batch.iter().flat_map(|j| part(j).iter().copied()).collect::<Vec<u8>>())
            };
            let (node, glob, adj, regions) =
                (join(|j| &j.node), join(|j| &j.glob), join(|j| &j.adj), join(|j| &j.regions));
            let out = evaluator
                .call_method1("evaluate", (count, node, glob, adj, regions))
                .unwrap_or_else(|e| panic!("PyTorch : {e}"));
            let (logits, value): (Bound<PyBytes>, Bound<PyBytes>) = out.extract().expect("logits et valeurs");
            let floats =
                |b: &[u8]| -> Vec<f32> { b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect() };
            let (logits, value) = (floats(logits.as_bytes()), floats(value.as_bytes()));
            let mut at = 0;
            for job in batch {
                let evals = (0..job.count)
                    .map(|k| {
                        let i = at + k;
                        let v: [f32; 10] = value[i * 10..(i + 1) * 10].try_into().expect("10 valeurs");
                        Eval::from_heads(job.players[k] as usize, logits[i * KEYS..(i + 1) * KEYS].to_vec(), &v)
                    })
                    .collect();
                at += job.count;
                // un fil de jeu arrêté ne relit pas : rien à faire
                let _ = job.reply.send(evals);
            }
        }
    });
}

#[cfg(not(feature = "gpu"))]
fn serve(_: &Path, _: mpsc::Receiver<Job>, ready: mpsc::Sender<Result<String, String>>) {
    let _ = ready.send(Err("programme lab3 compilé sans le GPU (cargo build --features gpu)".into()));
}
