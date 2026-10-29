//! La forme compacte des positions de l'auto-apprentissage, écrite par le selfplay : ce que lit
//! l'apprentissage (`lab11/py/data.py`, `Packed`), sans passer par les enregistrements pleins de
//! `selfplay::RECORD` (6 210 octets par position, ~900 ici). Un dossier par fil
//! (`compact_dir`), un fichier `.npy` par tableau, mappés tels quels par numpy :
//!
//! ```text
//! par position     players, regions (u8), weight (f32), win, gain, value, luck (f32 × 5),
//!                  adj_block (u32 : son bloc d'adjacences), nz_count, nz_big_count, pol_count (u16),
//!                  nz_glob (u8 × 103 : le masque des octets non nuls de la partie)
//! par région       owner (u8), nz_rows (u8 × 10 : le masque des octets non nuls de la région)
//! par bloc         adj_count (u8 : ses régions) ; adj (u64 par région : ses voisines)
//! par non nul      nz_ones (un bit, à la suite : vaut-il 1 ?) ; nz_big (u8 : les autres valeurs)
//! par coup         pol_key, pol_val (u16 : la politique des seuls coups possibles)
//! ```
//!
//! Les non nuls se suivent ligne par ligne : les régions présentes, puis la partie. Les masques
//! suivent `numpy.packbits` (le premier bit en tête de chaque octet). Cette forme redonne exactement
//! les champs des enregistrements pleins (`lab11/py/test_train.py` la compare aux `.pos` des mêmes
//! parties) ; la clé de chaque partie est calculée à la lecture.

use std::io::Write;
use std::path::{Path, PathBuf};

use smallmobile_kernel::ids::MAX_PLAYERS;

use crate::encode::{GLOBAL, NODE};
use crate::moves::Key;

/// Octets du masque d'une région, et de celui de la partie.
const ROW_MASK: usize = NODE.div_ceil(8);
const GLOBAL_MASK: usize = GLOBAL.div_ceil(8);

/// Une position, avec ses cibles.
pub struct Position<'a> {
    pub players: u8,
    pub weight: f32,
    pub win: [f32; MAX_PLAYERS],
    pub gain: [f32; MAX_PLAYERS],
    pub value: [f32; MAX_PLAYERS],
    /// La part du hasard dans `win` (`luck`), 0 sans `selfplay --luck`.
    pub luck: [f32; MAX_PLAYERS],
    /// À qui appartient chaque région présente (`selfplay::owners`).
    pub owners: &'a [u8],
    /// Les entrées des régions présentes, puis de la partie (`encode`).
    pub node: &'a [[u8; NODE]],
    pub global: &'a [u8; GLOBAL],
    pub policy: &'a [(Key, u16)],
}

/// Les positions d'une partie, ou de toutes celles d'un fil (`append`), sous forme compacte.
#[derive(Default)]
pub struct Compact {
    players: Vec<u8>,
    regions: Vec<u8>,
    weight: Vec<f32>,
    win: Vec<[f32; MAX_PLAYERS]>,
    gain: Vec<[f32; MAX_PLAYERS]>,
    value: Vec<[f32; MAX_PLAYERS]>,
    luck: Vec<[f32; MAX_PLAYERS]>,
    adj_block: Vec<u32>,
    nz_count: Vec<u16>,
    nz_big_count: Vec<u16>,
    pol_count: Vec<u16>,
    nz_glob: Vec<u8>,
    owner: Vec<u8>,
    nz_rows: Vec<u8>,
    adj_count: Vec<u8>,
    adj: Vec<u64>,
    /// Un octet par non nul (1 : il vaut 1), mis en bits à l'écriture.
    nz_ones: Vec<u8>,
    nz_big: Vec<u8>,
    pol_key: Vec<u16>,
    pol_val: Vec<u16>,
}

impl Compact {
    pub fn len(&self) -> usize {
        self.players.len()
    }

    pub fn is_empty(&self) -> bool {
        self.players.is_empty()
    }

    /// La part du hasard dans le résultat de chaque position (joueurs relatifs, `Position::luck`).
    pub fn luck(&self) -> &[[f32; MAX_PLAYERS]] {
        &self.luck
    }

    /// Ajoute une position ; `adj` : les voisines de chaque région de sa carte (un bit par région).
    pub fn push(&mut self, p: &Position, adj: &[u64]) {
        let regions = p.node.len();
        assert!(p.owners.len() == regions && adj.len() >= regions, "régions de la position");
        // la carte d'une partie ne change pas : un seul bloc tant qu'elle est la même
        let last = self.adj_count.len().checked_sub(1);
        let same = last.is_some_and(|b| {
            let start = self.adj.len() - self.adj_count[b] as usize;
            self.adj_count[b] as usize == regions && self.adj[start..] == adj[..regions]
        });
        if !same {
            self.adj_count.push(regions as u8);
            self.adj.extend_from_slice(&adj[..regions]);
        }
        self.adj_block.push((self.adj_count.len() - 1) as u32);
        self.players.push(p.players);
        self.regions.push(regions as u8);
        self.weight.push(p.weight);
        self.win.push(p.win);
        self.gain.push(p.gain);
        self.value.push(p.value);
        self.luck.push(p.luck);
        self.owner.extend_from_slice(p.owners);
        let (mut count, mut big) = (0u16, 0u16);
        let mut values = |bytes: &[u8], mask: &mut Vec<u8>| {
            let at = mask.len();
            mask.resize(at + bytes.len().div_ceil(8), 0);
            for (i, &v) in bytes.iter().enumerate() {
                if v != 0 {
                    mask[at + i / 8] |= 0x80 >> (i % 8);
                    count += 1;
                    self.nz_ones.push((v == 1) as u8);
                    if v != 1 {
                        big += 1;
                        self.nz_big.push(v);
                    }
                }
            }
        };
        for row in p.node {
            values(row, &mut self.nz_rows);
        }
        values(p.global, &mut self.nz_glob);
        self.nz_count.push(count);
        self.nz_big_count.push(big);
        self.pol_count.push(p.policy.len() as u16);
        for &(key, v) in p.policy {
            self.pol_key.push(key);
            self.pol_val.push(v);
        }
    }

    /// Ajoute les positions de `other` à la suite.
    pub fn append(&mut self, other: Compact) {
        let blocks = self.adj_count.len() as u32;
        self.adj_block.extend(other.adj_block.iter().map(|b| b + blocks));
        self.players.extend(other.players);
        self.regions.extend(other.regions);
        self.weight.extend(other.weight);
        self.win.extend(other.win);
        self.gain.extend(other.gain);
        self.value.extend(other.value);
        self.luck.extend(other.luck);
        self.nz_count.extend(other.nz_count);
        self.nz_big_count.extend(other.nz_big_count);
        self.pol_count.extend(other.pol_count);
        self.nz_glob.extend(other.nz_glob);
        self.owner.extend(other.owner);
        self.nz_rows.extend(other.nz_rows);
        self.adj_count.extend(other.adj_count);
        self.adj.extend(other.adj);
        self.nz_ones.extend(other.nz_ones);
        self.nz_big.extend(other.nz_big);
        self.pol_key.extend(other.pol_key);
        self.pol_val.extend(other.pol_val);
    }

    /// Écrit les tableaux dans `dir` (créé), un `.npy` chacun.
    pub fn write(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let n = self.len();
        let bytes = |v: &[u8]| v.to_vec();
        let le = |v: &mut dyn Iterator<Item = [u8; 4]>| v.flatten().collect::<Vec<u8>>();
        let f32s = |v: &[[f32; MAX_PLAYERS]]| le(&mut v.iter().flatten().map(|x| x.to_le_bytes()));
        let u16s = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let mut ones = vec![0u8; self.nz_ones.len().div_ceil(8)];
        for (i, _) in self.nz_ones.iter().enumerate().filter(|(_, o)| **o == 1) {
            ones[i / 8] |= 0x80 >> (i % 8);
        }
        let rows = self.owner.len();
        let arrays: [(&str, &str, Vec<usize>, Vec<u8>); 20] = [
            ("players", "|u1", vec![n], bytes(&self.players)),
            ("regions", "|u1", vec![n], bytes(&self.regions)),
            ("weight", "<f4", vec![n], le(&mut self.weight.iter().map(|x| x.to_le_bytes()))),
            ("win", "<f4", vec![n, MAX_PLAYERS], f32s(&self.win)),
            ("gain", "<f4", vec![n, MAX_PLAYERS], f32s(&self.gain)),
            ("value", "<f4", vec![n, MAX_PLAYERS], f32s(&self.value)),
            ("luck", "<f4", vec![n, MAX_PLAYERS], f32s(&self.luck)),
            ("adj_block", "<u4", vec![n], le(&mut self.adj_block.iter().map(|x| x.to_le_bytes()))),
            ("nz_count", "<u2", vec![n], u16s(&self.nz_count)),
            ("nz_big_count", "<u2", vec![n], u16s(&self.nz_big_count)),
            ("pol_count", "<u2", vec![n], u16s(&self.pol_count)),
            ("nz_glob", "|u1", vec![n, GLOBAL_MASK], bytes(&self.nz_glob)),
            ("owner", "|u1", vec![rows], bytes(&self.owner)),
            ("nz_rows", "|u1", vec![rows, ROW_MASK], bytes(&self.nz_rows)),
            ("adj_count", "|u1", vec![self.adj_count.len()], bytes(&self.adj_count)),
            ("adj", "<u8", vec![self.adj.len()], self.adj.iter().flat_map(|x| x.to_le_bytes()).collect()),
            ("nz_ones", "|u1", vec![ones.len()], ones),
            ("nz_big", "|u1", vec![self.nz_big.len()], bytes(&self.nz_big)),
            ("pol_key", "<u2", vec![self.pol_key.len()], u16s(&self.pol_key)),
            ("pol_val", "<u2", vec![self.pol_val.len()], u16s(&self.pol_val)),
        ];
        for (name, descr, shape, data) in arrays {
            npy(&dir.join(format!("{name}.npy")), descr, &shape, &data)?;
        }
        Ok(())
    }
}

/// Le dossier de la forme compacte d'un fichier de positions : `x.pos.compact` (`COMPACT` dans
/// `lab11/py/data.py`), à côté de `x.pos.games` et `x.pos.moves`.
pub fn compact_dir(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".compact");
    name.into()
}

/// Un tableau au format `.npy` (version 1.0) : l'en-tête, puis les octets.
fn npy(path: &Path, descr: &str, shape: &[usize], data: &[u8]) -> std::io::Result<()> {
    let dims = match shape {
        [n] => format!("({n},)"),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {dims}, }}");
    // magie, version et longueur (10 octets), puis l'en-tête fini par un saut de ligne : un multiple
    // de 64 octets
    header.push_str(&" ".repeat(63 - (10 + header.len()) % 64));
    header.push('\n');
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    f.write_all(data)?;
    f.flush()
}
