//! Le réseau de lab8 comme joueur des arènes de lab7 : il écrit ses plans avec le décodeur, sans
//! recherche (`plan:net=…[,t=…]`, température 0 par défaut : le coup le plus probable).

use std::path::Path;
use std::sync::Arc;

use smallmobile_kernel::{Action, Game};
use smallmobile_lab7::agent::Agent;

use crate::know::Knowledge;
use crate::moves;
use crate::net::Net;
use crate::plan::{Writer, pick};
use crate::rand::Rng;
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
