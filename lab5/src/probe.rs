//! Diagnostic des coups : positions indépendantes, sélection, puis confirmation indépendante.
//! Le cerveau de lab3 est inchangé. Toutes les suites jouent jusqu'au vrai résultat du jeu.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};
use std::time::Instant;

use smallmobile_kernel::{Game, Phase};

use crate::az::{Az, Evals, Oracle, Searched};
use crate::batch::Batch;
use crate::gpu::Gpu;
use crate::know::{Knowledge, determinize};
use crate::moves::{self, Choice, Key, Move};
use crate::rand::{Rng, mix};
use crate::table::{Outcome, Table};

const MAX_MOVES: usize = 20_000;
const BANDS: [&str; 3] = ["early", "middle", "late"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Positions,
    Select,
    Confirm,
}

impl Stage {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "positions" => Some(Self::Positions),
            "select" => Some(Self::Select),
            "confirm" => Some(Self::Confirm),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Positions => "positions",
            Self::Select => "select",
            Self::Confirm => "confirm",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub players: Vec<usize>,
    /// Positions par nombre de joueurs ET tiers de partie ; une partie indépendante par position.
    pub positions: usize,
    pub candidates: usize,
    pub select: usize,
    pub confirm: usize,
    pub seed: u64,
    pub threads: usize,
    pub concurrent: usize,
    pub out: PathBuf,
}

impl Config {
    pub fn validate(&self) -> io::Result<()> {
        if self.players.is_empty()
            || self.players.iter().any(|n| !(2..=5).contains(n))
            || self.players.iter().collect::<HashSet<_>>().len() != self.players.len()
            || self.positions == 0
            || !(4..=moves::KEYS).contains(&self.candidates)
            || self.select < 2
            || self.confirm < 2
            || self.threads == 0
            || self.concurrent == 0
        {
            return Err(io::Error::other(
                "paramètres invalides : joueurs distincts 2..5, positions/fils/concurrence > 0, candidats 4..608, suites >= 2",
            ));
        }
        self.positions.checked_mul(self.players.len() * 3).ok_or_else(|| io::Error::other("trop de positions"))?;
        self.candidates.checked_mul(self.select).ok_or_else(|| io::Error::other("trop de suites"))?;
        self.confirm.checked_mul(2).ok_or_else(|| io::Error::other("trop de confirmations"))?;
        Ok(())
    }
    pub fn total(&self) -> usize {
        self.positions * self.players.len() * 3
    }
    fn slot(&self, id: usize) -> (usize, usize) {
        (self.players[id % self.players.len()], (id / self.players.len()) % 3)
    }
    fn path(&self, id: usize, kind: &str) -> PathBuf {
        self.out.join(format!("p{id:06}.{kind}.tsv"))
    }
}

/// Graines séparées par position, rôle et suite. Aucune clé de coup : les alternatives sont appariées.
pub fn seeds(seed: u64, id: usize, stage: Stage, sample: usize) -> (u64, u64) {
    let domain = match stage {
        Stage::Positions => 0x706f736974696f6e,
        Stage::Select => 0x73656c656374,
        Stage::Confirm => 0x636f6e6669726d,
    };
    let base = mix(mix(mix(seed, id as u64), domain), sample as u64);
    (mix(base, 0), mix(base, 1))
}

fn atomic(path: &Path, text: &str) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = File::create(&tmp)?;
    f.write_all(text.as_bytes())?;
    f.sync_all()?;
    fs::rename(tmp, path)
}

/// Verrou libéré aussi à l'interruption du processus : deux commandes ne peuvent écrire le même lot.
fn lock(dir: &Path) -> io::Result<File> {
    unsafe extern "C" {
        fn flock(fd: std::ffi::c_int, operation: std::ffi::c_int) -> std::ffi::c_int;
    }
    let f = File::options().create(true).truncate(false).write(true).open(dir.join(".lock"))?;
    // SAFETY : descripteur ouvert, LOCK_EX | LOCK_NB, aucun pointeur.
    if unsafe { flock(f.as_raw_fd(), 2 | 4) } != 0 {
        return Err(io::Error::other("ce lot est déjà utilisé par une autre commande"));
    }
    Ok(f)
}

fn hash(path: &Path) -> io::Result<String> {
    let output = std::process::Command::new("timeout").args(["15", "sha256sum", "--"]).arg(path).output()?;
    if !output.status.success() {
        return Err(io::Error::other("échec sha256sum"));
    }
    let text = String::from_utf8(output.stdout).map_err(io::Error::other)?;
    Ok(text.split_whitespace().next().ok_or_else(|| io::Error::other("empreinte absente"))?.to_string())
}

/// Identité du lot : ni le nom du fichier de poids ni la machine ne doivent changer son contenu.
/// Changer de code, de poids, de backend ou de paramètres demande un autre dossier.
fn manifest(cfg: &Config, az: &Az, net: &Path, device: &str) -> io::Result<()> {
    let mut backend = format!("{device}:{}", std::env::consts::ARCH);
    #[cfg(target_arch = "x86_64")]
    if device == "cpu" {
        backend += &format!(
            ":avx2={}:fma={}:avx512f={}:avx512bf16={}",
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma"),
            std::is_x86_feature_detected!("avx512f"),
            std::is_x86_feature_detected!("avx512bf16")
        );
    }
    let text = format!(
        "format\tlab5-probe-1\nsource_sha256\t{}\nnet_sha256\t{}\nplayers\t{}\npositions_per_cell\t{}\ncandidates\t{}\nselect\t{}\nconfirm\t{}\nseed\t{}\nsimulations\t{}\ndevice\t{}\n",
        env!("LAB5_SOURCE_SHA256"),
        hash(net)?,
        cfg.players.iter().map(usize::to_string).collect::<Vec<_>>().join(","),
        cfg.positions,
        cfg.candidates,
        cfg.select,
        cfg.confirm,
        cfg.seed,
        az.cfg.iterations,
        backend,
    );
    let path = cfg.out.join("manifest.tsv");
    if path.exists() {
        if fs::read_to_string(path)? != text {
            return Err(io::Error::other(
                "lot incompatible : code, poids, backend ou paramètres différents ; choisir un autre --out",
            ));
        }
    } else {
        if fs::read_dir(&cfg.out)?.filter_map(Result::ok).any(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with('p')
                && [".position.tsv", ".select.tsv", ".confirm.tsv"].iter().any(|end| name.ends_with(end))
        }) {
            return Err(io::Error::other("résultats présents sans manifeste"));
        }
        atomic(&path, &text)?;
    }
    Ok(())
}

/// Une position est rejouable depuis sa carte, sa graine et la suite de clés de coups. L'histoire
/// publique et le redéploiement sont reconstruits aussi ; aucun état de Knowledge n'est perdu.
#[derive(Clone, Debug)]
pub struct Position {
    pub id: usize,
    pub players: usize,
    pub band: usize,
    pub map_seed: u64,
    pub game_seed: u32,
    pub round: i32,
    pub actor: usize,
    pub phase: String,
    pub baseline: Key,
    pub candidates: Vec<Key>,
    pub trace: Vec<Key>,
    pub descriptions: Vec<(Key, String)>,
}

fn list<T: std::fmt::Display>(xs: &[T]) -> String {
    xs.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}
fn number<T: std::str::FromStr>(s: &str) -> io::Result<T> {
    s.parse().map_err(|_| io::Error::other(format!("nombre invalide : {s}")))
}
fn numbers<T: std::str::FromStr>(s: &str) -> io::Result<Vec<T>> {
    if s.is_empty() { Ok(Vec::new()) } else { s.split(',').map(number).collect() }
}

impl Position {
    fn text(&self) -> String {
        let mut text = format!(
            "id\tplayers\tband\tmap_seed\tgame_seed\tround\tactor\tphase\tbaseline\tcandidates\ttrace\n{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            self.id,
            self.players,
            BANDS[self.band],
            self.map_seed,
            self.game_seed,
            self.round,
            self.actor,
            self.phase,
            self.baseline,
            list(&self.candidates),
            list(&self.trace)
        );
        for (key, desc) in &self.descriptions {
            text += &format!("# choice\t{key}\t{desc}\n");
        }
        text
    }
    pub fn read(path: &Path) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let f: Vec<_> =
            text.lines().nth(1).ok_or_else(|| io::Error::other("position incomplète"))?.split('\t').collect();
        if f.len() != 11 {
            return Err(io::Error::other("position mal formée"));
        }
        let band = BANDS.iter().position(|&s| s == f[2]).ok_or_else(|| io::Error::other("tiers inconnu"))?;
        let p = Position {
            id: number(f[0])?,
            players: number(f[1])?,
            band,
            map_seed: number(f[3])?,
            game_seed: number(f[4])?,
            round: number(f[5])?,
            actor: number(f[6])?,
            phase: f[7].into(),
            baseline: number(f[8])?,
            candidates: numbers(f[9])?,
            trace: numbers(f[10])?,
            descriptions: Vec::new(),
        };
        if !(2..=5).contains(&p.players)
            || p.actor >= p.players
            || p.candidates.len() < 2
            || !p.candidates.contains(&p.baseline)
            || p.candidates.iter().collect::<HashSet<_>>().len() != p.candidates.len()
        {
            return Err(io::Error::other("position incohérente"));
        }
        Ok(p)
    }
    pub fn replay<'m>(&self, table: &'m Table) -> io::Result<(Game<'m>, Knowledge)> {
        let (mut g, mut k) = (table.game(self.game_seed), Knowledge::default());
        for &key in &self.trace {
            play_key(&mut g, &mut k, key)?;
        }
        let legal = moves::legal(&g, &k);
        if g.state.round != self.round
            || g.state.actor() != Some(self.actor)
            || format!("{:?}", g.state.phase) != self.phase
            || self.candidates.iter().any(|key| !legal.iter().any(|c| c.key == *key))
        {
            return Err(io::Error::other("relecture de position différente"));
        }
        Ok((g, k))
    }
}

fn play_key(g: &mut Game<'_>, k: &mut Knowledge, key: Key) -> io::Result<()> {
    let mv = moves::legal(g, k)
        .into_iter()
        .find(|c| c.key == key)
        .ok_or_else(|| io::Error::other(format!("coup {key} illégal")))?
        .mv;
    k.play(g, &mv).map_err(|e| io::Error::other(format!("coup refusé : {e:?}")))
}

/// Le coup courant, déclin/fin des conquêtes, puis a priori diversifiés par victime et sorte de
/// coup. La dernière place disponible explore un autre coup au hasard. Ce petit ensemble n'est
/// PAS l'ensemble des plans possibles ; une absence de gain n'est pas une borne d'optimalité.
pub fn candidates(g: &Game<'_>, legal: &[Choice], s: &Searched, cap: usize, rng: &mut Rng) -> Vec<Key> {
    let baseline = legal.iter().find(|c| c.mv == s.mv).expect("coup légal").key;
    let prior = |key| s.details.iter().find(|d| d.0 == key).map_or(0.0, |d| d.1);
    let mut ranked = legal.to_vec();
    ranked.sort_by(|a, b| prior(b.key).total_cmp(&prior(a.key)).then(a.key.cmp(&b.key)));
    let mut out = vec![baseline];
    for c in &ranked {
        if matches!(c.mv, Move::Decline | Move::EndConquest) && !out.contains(&c.key) && out.len() < cap {
            out.push(c.key);
        }
    }
    let limit = cap.min(legal.len());
    let mut groups = HashSet::new();
    // Réserver une place à un coup qui peut être loin dans les a priori, seulement s'il faut tronquer.
    let top_limit = if legal.len() > cap { limit - 1 } else { limit };
    for c in &ranked {
        let group = match c.mv {
            Move::Region(kind, r) => format!(
                "{kind:?}/{:?}",
                g.state.regions[r as usize].occupant.race().map(|race| g.state.race(race).owner)
            ),
            other => format!("{other:?}"),
        };
        if out.len() < top_limit && !out.contains(&c.key) && groups.insert(group) {
            out.push(c.key);
        }
    }
    for c in &ranked {
        if out.len() < top_limit && !out.contains(&c.key) {
            out.push(c.key);
        }
    }
    if out.len() < limit {
        let rest: Vec<_> = ranked.iter().filter(|c| !out.contains(&c.key)).collect();
        out.push(rest[rng.below(rest.len())].key);
    }
    out
}

async fn sample(az: &Az, cfg: &Config, id: usize, table: &Table, oracle: &impl Oracle) -> io::Result<Position> {
    let (players, band) = cfg.slot(id);
    let (world, search) = seeds(cfg.seed, id, Stage::Positions, 0);
    let map_seed = world % 1_000_000_007;
    let game_seed = mix(world, 2) as u32;
    let mut sampling = Rng::new(mix(world, 3));
    let rounds: Vec<_> = (1..=table.rounds).filter(|r| ((*r - 1) * 3 / table.rounds) as usize == band).collect();
    let round = rounds[sampling.below(rounds.len())];
    let actor = sampling.below(players);
    let (mut g, mut k) = (table.game(game_seed), Knowledge::default());
    let mut trace = Vec::new();
    let mut evals = Evals::default();
    // Réservoir par phase pour un joueur et une manche tirés à l'avance ; les jetons ne prennent
    // pas toutes les places. Une défense hors tour peut donc aussi être choisie.
    let mut seen = BTreeMap::<String, (usize, Game<'_>, Knowledge, Vec<Key>)>::new();
    while g.state.phase != Phase::GameOver && g.state.round <= round {
        if trace.len() >= MAX_MOVES {
            return Err(io::Error::other("partie source trop longue"));
        }
        let legal = moves::legal(&g, &k);
        if g.state.round == round && g.state.actor() == Some(actor) && legal.len() > 1 {
            let phase = format!("{:?}", g.state.phase);
            let entry = seen.entry(phase).or_insert((0, g, k, trace.clone()));
            entry.0 += 1;
            if sampling.below(entry.0) == 0 {
                *entry = (entry.0, g, k, trace.clone());
            }
        }
        let mv = if legal.len() == 1 {
            legal[0].mv
        } else {
            az.run(&g, &k, &mut Rng::new(mix(search, trace.len() as u64)), az.cfg.iterations, false, &mut evals, oracle)
                .await
                .mv
        };
        trace.push(legal.iter().find(|c| c.mv == mv).expect("coup légal").key);
        k.play(&mut g, &mv).expect("coup de c130");
    }
    if seen.is_empty() {
        return Err(io::Error::other(format!(
            "position {id} : aucun choix dans le tour tiré ; conserver cet échec, ne pas remplacer discrètement la partie"
        )));
    }
    let selected_phase = sampling.below(seen.len());
    let (_, root, knowledge, trace) = seen.into_values().nth(selected_phase).expect("phase tirée");
    let legal = moves::legal(&root, &knowledge);
    let s = az
        .run(
            &root,
            &knowledge,
            &mut Rng::new(mix(search, u64::MAX)),
            az.cfg.iterations,
            false,
            &mut Evals::default(),
            oracle,
        )
        .await;
    let baseline = legal.iter().find(|c| c.mv == s.mv).expect("coup légal").key;
    let chosen = candidates(&root, &legal, &s, cfg.candidates, &mut sampling);
    let descriptions = legal
        .iter()
        .map(|c| {
            let (_, prior, q) = s.details.iter().find(|d| d.0 == c.key).copied().expect("détail de chaque coup");
            (c.key, format!("{} ; prior={prior:.6} ; search_value={q:.6}", moves::describe(&root, &c.mv)))
        })
        .collect();
    Ok(Position {
        id,
        players,
        band,
        map_seed,
        game_seed,
        round,
        actor,
        phase: format!("{:?}", root.state.phase),
        baseline,
        candidates: chosen,
        trace,
        descriptions,
    })
}

/// Un seul coup est changé ; ensuite tous les joueurs continuent comme c130. Le vrai monde n'est
/// jamais communiqué à la recherche, qui le redétermine à chaque simulation comme dans lab3.
pub async fn rollout(
    az: &Az,
    root: &Game<'_>,
    knowledge: &Knowledge,
    key: Key,
    world: u64,
    search: u64,
    oracle: &impl Oracle,
) -> io::Result<Outcome> {
    let mut g = determinize(root, knowledge, &mut Rng::new(world));
    let mut k = *knowledge;
    play_key(&mut g, &mut k, key)?;
    let mut evals = Evals::default();
    for step in 0..MAX_MOVES {
        if g.state.phase == Phase::GameOver {
            return Ok(Outcome::of(&g));
        }
        let mv = az
            .run(&g, &k, &mut Rng::new(mix(search, step as u64)), az.cfg.iterations, false, &mut evals, oracle)
            .await
            .mv;
        k.play(&mut g, &mv).expect("suite légale");
    }
    Err(io::Error::other("suite trop longue"))
}

#[derive(Clone, Copy, Debug)]
pub struct Trial {
    pub sample: usize,
    pub key: Key,
    pub world: u64,
    pub win: f64,
    pub margin: i32,
}

/// Le témoin gagne les égalités ; sinon clé croissante. La confirmation ne choisit jamais de coup.
pub fn select(baseline: Key, keys: &[Key], rows: &[Trial]) -> Key {
    let total = |key| rows.iter().filter(|r| r.key == key).map(|r| r.win).sum::<f64>();
    let mut best = baseline;
    let mut ordered = keys.to_vec();
    ordered.sort_unstable();
    for key in ordered {
        if total(key) > total(best) {
            best = key;
        }
    }
    best
}

fn trial_text(rows: &[Trial], selected: Key) -> String {
    let mut text = format!("# selected\t{selected}\nsample\tkey\tworld\twin\tmargin\n");
    for r in rows {
        text += &format!("{}\t{}\t{}\t{}\t{}\n", r.sample, r.key, r.world, r.win, r.margin);
    }
    text
}

fn read_trials(
    path: &Path,
    cfg: &Config,
    id: usize,
    stage: Stage,
    keys: &[Key],
    count: usize,
) -> io::Result<(Key, Vec<Trial>)> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let selected = number(
        lines
            .next()
            .and_then(|l| l.strip_prefix("# selected\t"))
            .ok_or_else(|| io::Error::other("sélection absente"))?,
    )?;
    if lines.next() != Some("sample\tkey\tworld\twin\tmargin") {
        return Err(io::Error::other("en-tête de suites invalide"));
    }
    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    for line in lines {
        let f: Vec<_> = line.split('\t').collect();
        if f.len() != 5 {
            return Err(io::Error::other("suite mal formée"));
        }
        let r = Trial {
            sample: number(f[0])?,
            key: number(f[1])?,
            world: number(f[2])?,
            win: number(f[3])?,
            margin: number(f[4])?,
        };
        if r.sample >= count
            || !keys.contains(&r.key)
            || r.world != seeds(cfg.seed, id, stage, r.sample).0
            || !(0.0..=1.0).contains(&r.win)
            || !seen.insert((r.sample, r.key))
        {
            return Err(io::Error::other("suites dupliquées ou incompatibles"));
        }
        rows.push(r);
    }
    if rows.len() != keys.len() * count || !keys.contains(&selected) {
        return Err(io::Error::other("suites incomplètes"));
    }
    rows.sort_by_key(|r| (r.sample, r.key));
    Ok((selected, rows))
}

type Job<'a, T> = Pin<Box<dyn Future<Output = io::Result<T>> + 'a>>;

/// Ordonnanceur de lab3 : groupes CPU ou suites concurrentes sur un anneau GPU. Les résultats
/// sont remis dans l'ordre des tâches, indépendamment des fils et de la vitesse des suites.
fn execute<'a, T>(
    batch: &Batch<'_>,
    gpu: Option<&Gpu>,
    count: usize,
    concurrent: usize,
    mut job: impl FnMut(usize) -> Job<'a, T>,
) -> io::Result<Vec<T>> {
    let mut cx = Context::from_waker(Waker::noop());
    let mut slots: Vec<Option<(usize, Job<'a, T>)>> = (0..concurrent.min(count)).map(|_| None).collect();
    let mut results: Vec<Option<T>> = (0..count).map(|_| None).collect();
    let mut next = 0;
    let mut advance = |slot: &mut Option<(usize, Job<'a, T>)>| -> io::Result<bool> {
        loop {
            if slot.is_none() {
                if next == count {
                    return Ok(false);
                }
                *slot = Some((next, job(next)));
                next += 1;
            }
            let (id, future) = slot.as_mut().expect("tâche");
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(result) => {
                    results[*id] = Some(result?);
                    *slot = None;
                }
                Poll::Pending => return Ok(true),
            }
        }
    };
    if let Some(gpu) = gpu {
        let mut ready: Vec<_> = (0..slots.len()).map(|s| (s, None)).collect();
        let mut waiting = VecDeque::new();
        loop {
            while let Some((s, evals)) = ready.pop() {
                if let Some(evals) = evals {
                    batch.resume(evals);
                }
                if advance(&mut slots[s])? {
                    waiting.push_back((s, batch.send().expect("évaluations déposées")));
                }
            }
            let Some((s, sent)) = waiting.pop_front() else { break };
            ready.push((s, Some(batch.receive(sent))));
            while waiting.front().is_some_and(|(_, sent)| gpu.done(sent)) {
                let (s, sent) = waiting.pop_front().unwrap();
                ready.push((s, Some(batch.receive(sent))));
            }
        }
    } else {
        loop {
            for slot in &mut slots {
                advance(slot)?;
            }
            if slots.iter().all(Option::is_none) {
                break;
            }
            batch.submit();
        }
    }
    Ok(results.into_iter().map(|r| r.expect("résultat de chaque tâche")).collect())
}

fn work(az: &Az, cfg: &Config, stage: Stage, id: usize, batch: &Batch<'_>, gpu: Option<&Gpu>) -> io::Result<()> {
    let pos_path = cfg.path(id, "position");
    if stage == Stage::Positions && !pos_path.exists() {
        let table = Table::new(cfg.slot(id).0, seeds(cfg.seed, id, Stage::Positions, 0).0 % 1_000_000_007);
        let p = execute(batch, gpu, 1, 1, |_| Box::pin(sample(az, cfg, id, &table, batch)))?.remove(0);
        return atomic(&pos_path, &p.text());
    }
    let p = Position::read(&pos_path)?;
    let (players, band) = cfg.slot(id);
    let (world, _) = seeds(cfg.seed, id, Stage::Positions, 0);
    if (p.id, p.players, p.band, p.map_seed, p.game_seed)
        != (id, players, band, world % 1_000_000_007, mix(world, 2) as u32)
    {
        return Err(io::Error::other("position d'un autre lot"));
    }
    let table = Table::new(p.players, p.map_seed);
    let (g, k) = p.replay(&table)?;
    if stage == Stage::Positions {
        return Ok(());
    }
    let (keys, count, selected) = if stage == Stage::Select {
        (p.candidates.clone(), cfg.select, None)
    } else {
        let (selected, rows) = read_trials(&cfg.path(id, "select"), cfg, id, Stage::Select, &p.candidates, cfg.select)?;
        if select(p.baseline, &p.candidates, &rows) != selected {
            return Err(io::Error::other("sélection modifiée"));
        }
        (
            if selected == p.baseline { vec![p.baseline] } else { vec![p.baseline, selected] },
            cfg.confirm,
            Some(selected),
        )
    };
    let path = cfg.path(id, stage.name());
    if path.exists() {
        let (saved, rows) = read_trials(&path, cfg, id, stage, &keys, count)?;
        if saved != selected.unwrap_or_else(|| select(p.baseline, &keys, &rows)) {
            return Err(io::Error::other("résultat incompatible"));
        }
        return Ok(());
    }
    let rows = execute(batch, gpu, keys.len() * count, cfg.concurrent, |i| {
        let (sample, key) = (i / keys.len(), keys[i % keys.len()]);
        let (world, search) = seeds(cfg.seed, id, stage, sample);
        let (g, k, actor) = (&g, &k, p.actor);
        Box::pin(async move {
            let o = rollout(az, g, k, key, world, search, batch).await?;
            Ok(Trial { sample, key, world, win: o.win[actor], margin: o.margin(actor) })
        })
    })?;
    atomic(&path, &trial_text(&rows, selected.unwrap_or_else(|| select(p.baseline, &keys, &rows))))
}

/// Une seule étape par invocation. Les fichiers atomiques terminés sont validés et réutilisés.
pub fn run(az: &Az, cfg: &Config, stage: Stage, net: &Path, device: &str, gpu: Option<&Gpu>) -> io::Result<()> {
    cfg.validate()?;
    if az.cfg.iterations == 0 {
        return Err(io::Error::other("simulations > 0"));
    }
    if std::env::var_os("LAB5_INSTANT").is_some() {
        return Err(io::Error::other("LAB5_INSTANT interdit pour ce diagnostic"));
    }
    fs::create_dir_all(&cfg.out)?;
    let _lock = lock(&cfg.out)?;
    manifest(cfg, az, net, device)?;
    // Vérifier TOUS les prérequis avant de lancer la moindre suite.
    for id in 0..cfg.total() {
        if stage != Stage::Positions && !cfg.path(id, "position").exists() {
            return Err(io::Error::other("étape positions incomplète"));
        }
        if stage == Stage::Confirm && !cfg.path(id, "select").exists() {
            return Err(io::Error::other("étape select incomplète"));
        }
    }
    let start = Instant::now();
    let threads = cfg.threads.min(cfg.total());
    let lanes = gpu.map(|g| g.lanes(threads)).unwrap_or_default();
    if lanes.first().is_some_and(|l| cfg.concurrent.saturating_mul(crate::az::CONSIDERED) > l.free()) {
        return Err(io::Error::other("concurrence trop grande pour l'anneau GPU"));
    }
    let mut invocations = File::options().create(true).append(true).open(cfg.out.join("invocations.tsv"))?;
    writeln!(
        invocations,
        "start\t{}\t{}\tthreads={threads}\tconcurrent={}",
        stage.name(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        cfg.concurrent
    )?;
    let (next, done) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let result = std::thread::scope(|scope| {
        let mut lanes = lanes.into_iter();
        let mut handles = Vec::new();
        for _ in 0..threads {
            let lane = lanes.next();
            let (next, done) = (&next, &done);
            handles.push(scope.spawn(move || -> io::Result<()> {
                let batch = Batch::new(&az.net, gpu.zip(lane).map(|(g, l)| (g, l, 0)));
                loop {
                    let id = next.fetch_add(1, Ordering::Relaxed);
                    if id >= cfg.total() {
                        return Ok(());
                    }
                    let position_start = Instant::now();
                    if let Err(e) = work(az, cfg, stage, id, &batch, gpu) {
                        next.store(cfg.total(), Ordering::Relaxed);
                        return Err(e);
                    }
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!(
                        "{} : {n}/{} positions, p{id:06} ({} j., {}) en {:.1} s ; total {:.1} s",
                        stage.name(),
                        cfg.total(),
                        cfg.slot(id).0,
                        BANDS[cfg.slot(id).1],
                        position_start.elapsed().as_secs_f64(),
                        start.elapsed().as_secs_f64()
                    );
                }
            }));
        }
        for h in handles {
            h.join().map_err(|_| io::Error::other("fil interrompu"))??;
        }
        Ok(())
    });
    writeln!(
        invocations,
        "end\t{}\t{:.3}\t{}",
        stage.name(),
        start.elapsed().as_secs_f64(),
        if result.is_ok() { "ok" } else { "error" }
    )?;
    result
}

/// Relecture sans réseau ni simulation, pour examiner une occasion manquée signalée par Python.
pub fn inspect(dir: &Path, id: usize) -> io::Result<String> {
    let path = dir.join(format!("p{id:06}.position.tsv"));
    let p = Position::read(&path)?;
    let table = Table::new(p.players, p.map_seed);
    let (g, k) = p.replay(&table)?;
    let mut text = format!(
        "p{id:06}, {} joueurs, {}, manche {}/{}, joueur {}, {}\nTémoin : clé {} ; candidats : {}\n",
        p.players,
        BANDS[p.band],
        p.round,
        g.state.max_rounds,
        p.actor + 1,
        p.phase,
        p.baseline,
        list(&p.candidates)
    );
    for (q, player) in g.state.players.iter().enumerate() {
        text += &format!("joueur {} : {player:?}\n", q + 1);
    }
    for (r, region) in g.state.regions.iter().enumerate().take(g.map.len()) {
        text +=
            &format!("région {r} ({:?}, voisines {:?}) : {region:?}\n", g.map.regions[r].terrain, g.map.regions[r].adj);
    }
    text += "Coups légaux :\n";
    for c in moves::legal(&g, &k) {
        text += &format!(
            "{}{} : {}\n",
            c.key,
            if p.candidates.contains(&c.key) { " *" } else { "" },
            moves::describe(&g, &c.mv)
        );
    }
    text += "Évaluation initiale (valeur mêlant victoire et pièces, non fréquence observée) :\n";
    for line in fs::read_to_string(path)?.lines().filter(|l| l.starts_with("# choice\t")) {
        text += line;
        text.push('\n');
    }
    Ok(text)
}
