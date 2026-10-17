//! Les modèles appris de l'IA : des sommes de petits arbres de décision (gradient boosting). Le
//! noyau ne fait que les évaluer.
//!
//! Format des fichiers `models/*.bin` (petit-boutiste) :
//! `"SMB1"`, nombre d'entrées (u16) et leurs noms (u8 + UTF-8), valeur de base (f64), nombre
//! d'arbres (u32), puis pour chaque arbre son nombre de nœuds (u32) et ses nœuds : entrée (i16, −1
//! pour une feuille), seuil ou valeur (f64), enfant « oui » (u16), enfant « non » (u16). « Oui » :
//! entrée ≤ seuil.

use std::sync::OnceLock;

use super::features::MAX_INPUTS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    Truncated,
    BadMagic,
    BadText,
    BadNode,
    TrailingBytes,
}

/// Un nœud intérieur d'arbre complet : la question « x[input] ≤ threshold ? ».
#[derive(Clone, Copy, Debug)]
struct Split {
    threshold: f64,
    input: u32,
}

/// Les arbres rendus complets et rangés en tas : le nœud i a pour enfants 2i + 1 (oui) et 2i + 2
/// (non), sans pointeurs. Une feuille atteinte avant la profondeur maximale est prolongée par des
/// questions toujours vraies. La somme se fait dans l'ordre des arbres, comme à l'apprentissage.
#[derive(Clone, Debug)]
pub struct Model {
    pub inputs: Vec<String>,
    base: f64,
    depth: u32,
    splits: Vec<Split>,
    leaves: Vec<f64>,
}

/// Arbres parcourus ensemble : leurs parcours sont indépendants, le processeur les mène de front.
const LANES: usize = 8;

/// Un nœud d'arbre tel qu'appris : question « x[input] ≤ value ? » (enfants `yes` et `no`), ou
/// feuille (`input` < 0) de valeur `value`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub input: i16,
    pub value: f64,
    pub yes: u16,
    pub no: u16,
}

#[cfg(test)]
impl Node {
    fn leaf(value: f64) -> Node {
        Node { input: -1, value, yes: 0, no: 0 }
    }
}

struct Bytes<'a>(&'a [u8]);

impl<'a> Bytes<'a> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], ModelError> {
        let (head, rest) = self.0.split_first_chunk::<N>().ok_or(ModelError::Truncated)?;
        self.0 = rest;
        Ok(*head)
    }
    fn u8(&mut self) -> Result<u8, ModelError> {
        Ok(self.take::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16, ModelError> {
        Ok(u16::from_le_bytes(self.take()?))
    }
    fn i16(&mut self) -> Result<i16, ModelError> {
        Ok(i16::from_le_bytes(self.take()?))
    }
    fn u32(&mut self) -> Result<u32, ModelError> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn f64(&mut self) -> Result<f64, ModelError> {
        Ok(f64::from_le_bytes(self.take()?))
    }
    fn text(&mut self) -> Result<String, ModelError> {
        let len = self.u8()? as usize;
        if self.0.len() < len {
            return Err(ModelError::Truncated);
        }
        let (head, rest) = self.0.split_at(len);
        self.0 = rest;
        String::from_utf8(head.to_vec()).map_err(|_| ModelError::BadText)
    }
}

impl Model {
    pub fn parse(bytes: &[u8]) -> Result<Model, ModelError> {
        let mut b = Bytes(bytes);
        if &b.take::<4>()? != b"SMB1" {
            return Err(ModelError::BadMagic);
        }
        let inputs = (0..b.u16()?).map(|_| b.text()).collect::<Result<Vec<_>, _>>()?;
        let base = b.f64()?;
        let mut trees = Vec::new();
        for _ in 0..b.u32()? {
            let nodes = (0..b.u32()?)
                .map(|_| Ok(Node { input: b.i16()?, value: b.f64()?, yes: b.u16()?, no: b.u16()? }))
                .collect::<Result<Vec<_>, ModelError>>()?;
            trees.push(nodes);
        }
        if !b.0.is_empty() {
            return Err(ModelError::TrailingBytes);
        }
        Self::compile(inputs, base, &trees)
    }

    fn compile(inputs: Vec<String>, base: f64, trees: &[Vec<Node>]) -> Result<Model, ModelError> {
        fn depth_of(tree: &[Node], n: usize, budget: u32) -> Result<u32, ModelError> {
            let node = tree.get(n).ok_or(ModelError::BadNode)?;
            if node.input < 0 {
                return Ok(0);
            }
            if budget == 0 || node.input as usize >= MAX_INPUTS {
                return Err(ModelError::BadNode);
            }
            let yes = depth_of(tree, node.yes as usize, budget - 1)?;
            let no = depth_of(tree, node.no as usize, budget - 1)?;
            Ok(1 + yes.max(no))
        }
        let mut depth = 1;
        for tree in trees {
            depth = depth.max(depth_of(tree, 0, 16)?);
        }
        let inner = (1usize << depth) - 1;
        let mut m = Model {
            inputs,
            base,
            depth,
            splits: vec![Split { threshold: f64::INFINITY, input: 0 }; trees.len() * inner],
            leaves: vec![0.0; trees.len() << depth],
        };
        for (t, tree) in trees.iter().enumerate() {
            m.fill(tree, t, 0, 0, 0);
        }
        if m.inputs.len() > MAX_INPUTS || m.splits.iter().any(|s| s.input as usize >= m.inputs.len()) {
            return Err(ModelError::BadNode);
        }
        Ok(m)
    }

    fn fill(&mut self, tree: &[Node], t: usize, n: usize, at: usize, level: u32) {
        let inner = (1usize << self.depth) - 1;
        let node = tree[n];
        if level == self.depth {
            self.leaves[(t << self.depth) + at - inner] = node.value;
            return;
        }
        let leaf = node.input < 0;
        if !leaf {
            self.splits[t * inner + at] = Split { threshold: node.value, input: node.input as u32 };
        }
        let (yes, no) = if leaf { (n, n) } else { (node.yes as usize, node.no as usize) };
        self.fill(tree, t, yes, 2 * at + 1, level + 1);
        self.fill(tree, t, no, 2 * at + 2, level + 1);
    }

    /// Valeur prédite pour les entrées `x`, dans l'ordre de `inputs`.
    pub fn predict(&self, x: &[f64]) -> f64 {
        let x = &x[..self.inputs.len()];
        let inner = (1usize << self.depth) - 1;
        let trees = self.leaves.len() >> self.depth;
        // « oui » si x ≤ seuil ; écrit `!(x > seuil)` pour qu'une entrée invalide (NaN) aille à « non »,
        // comme en JavaScript
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let step = |tree: usize, n: usize| {
            let s = self.splits[tree * inner + n];
            2 * n + 1 + usize::from(!(x[s.input as usize] <= s.threshold))
        };
        let mut v = self.base;
        let mut t = 0;
        while t + LANES <= trees {
            let mut n = [0usize; LANES];
            for _ in 0..self.depth {
                for (k, n) in n.iter_mut().enumerate() {
                    *n = step(t + k, *n);
                }
            }
            for (k, n) in n.iter().enumerate() {
                v += self.leaves[((t + k) << self.depth) + n - inner];
            }
            t += LANES;
        }
        for t in t..trees {
            let mut n = 0;
            for _ in 0..self.depth {
                n = step(t, n);
            }
            v += self.leaves[(t << self.depth) + n - inner];
        }
        v
    }

    /// Valeurs prédites pour plusieurs lignes d'entrées.
    pub fn predict_all<'a>(&self, rows: impl IntoIterator<Item = &'a [f64]>) -> Vec<f64> {
        rows.into_iter().map(|x| self.predict(x)).collect()
    }
}

/// Les modèles livrés avec le jeu.
#[derive(Clone)]
pub struct Models {
    pub pick: Model,
    pub entry: Model,
    pub conquest: Model,
    pub conquest_lead: Model,
    pub decline: Model,
    pub hold: Model,
}

impl Models {
    /// Les modèles intégrés au noyau, lus une fois.
    pub fn get() -> &'static Models {
        static MODELS: OnceLock<Models> = OnceLock::new();
        MODELS.get_or_init(|| {
            let load = |bytes: &[u8]| Model::parse(bytes).expect("modèle intégré illisible");
            let models = Models {
                pick: load(include_bytes!("../../models/pick.bin")),
                entry: load(include_bytes!("../../models/entry.bin")),
                conquest: load(include_bytes!("../../models/conquest.bin")),
                conquest_lead: load(include_bytes!("../../models/conquest-lead.bin")),
                decline: load(include_bytes!("../../models/decline.bin")),
                hold: load(include_bytes!("../../models/hold.bin")),
            };
            models.check().expect("modèle intégré incompatible");
            models
        })
    }

    /// Chaque modèle lit-il bien un début de la liste de ses entrées ?
    pub fn check(&self) -> Result<(), &'static str> {
        use super::features::{conquest_layout, decline_layout, entry_layout, hold_layout, pick_layout};
        let pairs = [
            (&self.pick, pick_layout(), "pick"),
            (&self.entry, entry_layout(), "entry"),
            (&self.conquest, conquest_layout(), "conquest"),
            (&self.conquest_lead, conquest_layout(), "conquest-lead"),
            (&self.decline, decline_layout(), "decline"),
            (&self.hold, hold_layout(), "hold"),
        ];
        for (model, layout, name) in pairs {
            if !layout.accepts(&model.inputs) {
                return Err(name);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(trees: &[Vec<Node>]) -> Model {
        Model::compile(vec!["a".into(), "b".into()], 0.5, trees).unwrap()
    }

    #[test]
    fn shipped_models_match_their_inputs() {
        assert_eq!(Models::get().check(), Ok(()));
    }

    #[test]
    fn evaluates_uneven_trees() {
        let leaf = Node::leaf;
        // arbre 1 : a ≤ 1 ? (b ≤ 0 ? 10 : 20) : 30 ; arbre 2 : une feuille
        let t1 = vec![
            Node { input: 0, value: 1.0, yes: 1, no: 2 },
            Node { input: 1, value: 0.0, yes: 3, no: 4 },
            leaf(30.0),
            leaf(10.0),
            leaf(20.0),
        ];
        let m = model(&[t1, vec![leaf(1.0)]]);
        assert_eq!(m.predict(&[1.0, 0.0]), 11.5);
        assert_eq!(m.predict(&[1.0, 2.0]), 21.5);
        assert_eq!(m.predict(&[3.0, 0.0]), 31.5);
        assert_eq!(m.predict(&[0.5, -1.0]), 11.5);
        let rows: [&[f64]; 3] = [&[1.0, 0.0], &[1.0, 2.0], &[3.0, 0.0]];
        assert_eq!(m.predict_all(rows), vec![11.5, 21.5, 31.5]);
    }
}
