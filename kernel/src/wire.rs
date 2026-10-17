//! Le protocole entre le noyau et TypeScript : des mots de 32 bits, petit-boutistes. TypeScript
//! l'écrit et le lit dans `src/engine/kernel.ts`. Les entiers signés sont écrits en complément à deux ; « aucun » vaut −1
//! (régions, joueurs) ou 255 (peuples, pouvoirs).
//!
//! Une partie (`decode_game`) : `MAGIC`, la carte, puis l'état.
//! - carte : graine (2 mots), nombre de régions, puis par région : terrain, drapeaux (caverne, magie,
//!   mine, entrée, bord, côte), éloignement du centre en dixièmes, nombre de voisines, voisines ;
//! - état : voir `encode_state`, qui l'écrit dans le même ordre.

use crate::action::{Action, Event, How, IncomeKind, IncomeLine, Logged, Placement};
use crate::collections::{ArrayVec, PlayerSet};
use crate::game::Game;
use crate::ids::{Level, MAX_PLAYERS, MAX_REGIONS, Phase, Power, Race, Terrain};
use crate::map::{Map, RegionDef};
use crate::reason::Reason;
use crate::rules::{AttackHint, AttackHow, Attacker, Modifier, Special, TargetKind, Via};
use crate::state::{Combo, DefenseTask, DieRoll, Occupant, Peace, Player, RaceState, RegionState, Stage, State, Turn};

pub const MAGIC: u32 = 0x534d_4b32; // « SMK2 »
const NONE_ID: u32 = 255;
const LOST_ID: u32 = 254;

/// Le message est mal formé ou décrit un état impossible (et pourquoi).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Malformed(pub &'static str);

/// Une valeur manquante ou hors limites.
const BAD: Malformed = Malformed("valeur manquante ou hors limites");

type Result<T> = core::result::Result<T, Malformed>;

pub struct Reader<'a> {
    words: &'a [u32],
    at: usize,
}

impl<'a> Reader<'a> {
    pub fn new(words: &'a [u32]) -> Self {
        Reader { words, at: 0 }
    }

    fn word(&mut self) -> Result<u32> {
        let w = *self.words.get(self.at).ok_or(BAD)?;
        self.at += 1;
        Ok(w)
    }

    /// Un entier de 0 à `max`.
    fn upto(&mut self, max: usize) -> Result<usize> {
        let w = self.word()? as usize;
        if w <= max { Ok(w) } else { Err(BAD) }
    }

    fn int(&mut self) -> Result<i32> {
        Ok(self.word()? as i32)
    }

    /// Un entier positif ou nul raisonnable (pièces, jetons, tours).
    fn count(&mut self) -> Result<i32> {
        let n = self.int()?;
        if (0..=1_000_000).contains(&n) { Ok(n) } else { Err(BAD) }
    }

    fn flag(&mut self) -> Result<bool> {
        Ok(self.upto(1)? == 1)
    }

    /// Un indice de 0 à `len − 1`, ou −1 pour « aucun ».
    fn index_or_none(&mut self, len: usize) -> Result<Option<usize>> {
        match self.int()? {
            -1 => Ok(None),
            i if i >= 0 && (i as usize) < len => Ok(Some(i as usize)),
            _ => Err(BAD),
        }
    }

    fn race(&mut self) -> Result<Race> {
        Race::from_index(self.word()? as usize).ok_or(BAD)
    }

    fn race_or_none(&mut self) -> Result<Option<Race>> {
        match self.word()? {
            NONE_ID => Ok(None),
            i => Race::from_index(i as usize).map(Some).ok_or(BAD),
        }
    }

    fn power(&mut self) -> Result<Power> {
        Power::from_index(self.word()? as usize).ok_or(BAD)
    }

    fn list<T: Copy + Default, const N: usize>(
        &mut self,
        mut item: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<ArrayVec<T, N>> {
        let len = self.upto(N)?;
        let mut out = ArrayVec::new();
        for _ in 0..len {
            out.try_push(item(self)?).map_err(|_| BAD)?;
        }
        Ok(out)
    }

    pub fn done(&self) -> Result<()> {
        if self.at == self.words.len() { Ok(()) } else { Err(BAD) }
    }
}

pub struct Writer<'a> {
    out: &'a mut [u32],
    at: usize,
}

impl<'a> Writer<'a> {
    pub fn new(out: &'a mut [u32]) -> Self {
        Writer { out, at: 0 }
    }

    pub fn word(&mut self, w: u32) -> &mut Self {
        assert!(self.at < self.out.len(), "tampon trop petit");
        self.out[self.at] = w;
        self.at += 1;
        self
    }

    pub fn int(&mut self, i: i32) -> &mut Self {
        self.word(i as u32)
    }

    fn opt(&mut self, i: Option<usize>) -> &mut Self {
        self.int(i.map_or(-1, |i| i as i32))
    }

    fn race(&mut self, r: Option<Race>) -> &mut Self {
        self.word(r.map_or(NONE_ID, |r| r.index() as u32))
    }

    pub fn len(&self) -> usize {
        self.at
    }

    /// Les mots écrits.
    pub fn written(&self) -> &[u32] {
        &self.out[..self.at]
    }

    pub fn is_empty(&self) -> bool {
        self.at == 0
    }
}

// --- carte et état -----------------------------------------------------------------------------

pub fn decode_map(r: &mut Reader) -> Result<Map> {
    let seed = r.word()? as u64 | (r.word()? as u64) << 32;
    let n = r.upto(MAX_REGIONS)?;
    let mut regions = Vec::with_capacity(n);
    for _ in 0..n {
        let terrain = Terrain::from_index(r.word()? as usize).ok_or(BAD)?;
        let flags = r.upto(63)?;
        let centre = r.upto(10)? as f64 / 10.0;
        let degree = r.upto(n)?;
        let adj = (0..degree).map(|_| r.upto(n - 1).map(|a| a as u8)).collect::<Result<Vec<_>>>()?;
        regions.push(RegionDef {
            terrain,
            cavern: flags & 1 != 0,
            magic: flags & 2 != 0,
            mine: flags & 4 != 0,
            entry: flags & 8 != 0,
            border: flags & 16 != 0,
            coastal: flags & 32 != 0,
            adj,
            centre,
        });
    }
    Map::new(seed, regions).map_err(|_| BAD)
}

pub fn encode_map(map: &Map, w: &mut Writer) {
    w.word(map.seed as u32).word((map.seed >> 32) as u32).word(map.len() as u32);
    for d in &map.regions {
        let flags = d.cavern as u32
            | (d.magic as u32) << 1
            | (d.mine as u32) << 2
            | (d.entry as u32) << 3
            | (d.border as u32) << 4
            | (d.coastal as u32) << 5;
        w.word(d.terrain.index() as u32).word(flags).word((d.centre * 10.0).round() as u32).word(d.adj.len() as u32);
        for &a in &d.adj {
            w.word(a as u32);
        }
    }
}

/// Une partie complète, comme la lit `decode_game`.
pub fn encode_game(map: &Map, state: &State, w: &mut Writer) {
    w.word(MAGIC);
    encode_map(map, w);
    encode_state(state, map.len(), w);
}

/// Une partie complète : la carte et l'état.
pub fn decode_game(words: &[u32]) -> Result<(Map, State)> {
    let mut r = Reader::new(words);
    if r.word()? != MAGIC {
        return Err(BAD);
    }
    let map = decode_map(&mut r)?;
    let state = decode_state(&mut r, &map)?;
    r.done()?;
    Ok((map, state))
}

pub fn decode_state(r: &mut Reader, map: &Map) -> Result<State> {
    let n = map.len();
    let count = r.upto(MAX_PLAYERS)?;
    if count < 2 {
        return Err(BAD);
    }
    let current = r.upto(count - 1)?;
    let phase = Phase::from_index(r.word()? as usize).ok_or(BAD)?;
    let round = r.count()?;
    let max_rounds = r.count()?;
    let seq = r.word()?;
    let rng = r.word()?;
    let last_region = r.index_or_none(n)?;
    let last_die = match r.flag()? {
        false => None,
        true => Some(DieRoll {
            seq: r.word()?,
            player: r.upto(count - 1)?,
            value: r.count()?,
            need: r.int()?,
            success: match r.int()? {
                -1 => None,
                0 => Some(false),
                1 => Some(true),
                _ => return Err(BAD),
            },
            region: r.index_or_none(n)?,
            berserk: r.flag()?,
        }),
    };
    let mut players = ArrayVec::new();
    for _ in 0..count {
        players.push(Player {
            level: Level::from_index(r.word()? as usize).ok_or(BAD)?,
            coins: r.int()?,
            active: r.race_or_none()?,
            last_income: r.int()?,
            declined: r.list(|r| r.race())?,
        });
    }
    let mut races = [None; Race::COUNT];
    for slot in races.iter_mut() {
        let power = r.word()?;
        let owner = r.upto(count - 1)?;
        let declined = r.flag()?;
        let hand = r.int()?;
        let holes_placed = r.upto(2)? as i32;
        let picked_round = r.count()?;
        if power != NONE_ID {
            let power = Power::from_index(power as usize).ok_or(BAD)?;
            *slot = Some(RaceState { power, owner, declined, hand, holes_placed, picked_round });
        }
    }
    let mut regions = [RegionState::default(); MAX_REGIONS];
    for region in regions.iter_mut().take(n) {
        let occupant = match r.word()? {
            NONE_ID => Occupant::Empty,
            LOST_ID => Occupant::Lost,
            i => Occupant::Race(Race::from_index(i as usize).ok_or(BAD)?),
        };
        let tokens = r.count()?;
        let flags = r.upto(31)?;
        let camps = r.count()?;
        *region = RegionState {
            occupant,
            tokens,
            camps,
            lair: flags & 1 != 0,
            fortress: flags & 2 != 0,
            hero: flags & 4 != 0,
            dragon: flags & 8 != 0,
            hole: flags & 16 != 0,
        };
    }
    let market = r.list(|r| Ok(Combo { race: r.race()?, power: r.power()?, coins: r.count()? }))?;
    let race_deck = r.list(|r| r.race())?;
    let power_deck = r.list(|r| r.power())?;
    let mut turn = Turn {
        stage: if r.flag()? { Stage::Ghouls } else { Stage::Active },
        army: r.race_or_none()?,
        conquests: r.count()?,
        non_empty: r.count()?,
        attacked: PlayerSet(r.upto((1 << count) - 1)? as u8),
        sorcery: PlayerSet(r.upto((1 << count) - 1)? as u8),
        dragon_used: r.flag()?,
        fortress_region: r.index_or_none(n)?,
        berserk_die: match r.int()? {
            -1 => None,
            d @ 0..=3 => Some(d),
            _ => return Err(BAD),
        },
        die_rolled: r.flag()?,
        declined: r.flag()?,
        picked: r.flag()?,
        amazon_bonus: r.count()?,
        readied: [0; MAX_REGIONS],
    };
    for readied in turn.readied.iter_mut().take(n) {
        *readied = r.count()?;
    }
    let defense = r.list(|r| Ok(DefenseTask { player: r.upto(count - 1)?, race: r.race()? }))?;
    let peace = r.list(|r| Ok(Peace { diplomat: r.upto(count - 1)?, target: r.upto(count - 1)? }))?;
    let state = State {
        players,
        races,
        regions,
        market,
        race_deck,
        power_deck,
        round,
        max_rounds,
        current,
        phase,
        turn,
        defense,
        peace,
        seq,
        rng,
        last_die,
        last_region,
    };
    validate(&state, map)?;
    Ok(state)
}

/// Les invariants dont dépend le moteur : un état qui les viole est refusé plutôt que de faire
/// planter le noyau plus tard.
fn validate(s: &State, map: &Map) -> Result<()> {
    let ok = |c: bool, why: &'static str| if c { Ok(()) } else { Err(Malformed(why)) };
    let in_play = |r: Race| s.races[r.index()].is_some();
    ok(s.max_rounds >= 1 && s.round >= 1 && s.round <= s.max_rounds, "manche impossible")?;
    if matches!(s.phase, Phase::Conquest | Phase::Redeploy) {
        ok(s.turn.army.is_some_and(in_play), "aucun peuple en action")?;
    }
    if s.phase == Phase::Defend {
        ok(!s.defense.is_empty(), "défense sans joueur à défendre")?;
    }
    for region in &s.regions[..map.len()] {
        if let Occupant::Race(r) = region.occupant {
            ok(in_play(r), "région occupée par un peuple absent")?;
        }
    }
    for (pid, p) in s.players.iter().enumerate() {
        if let Some(a) = p.active {
            ok(
                s.races[a.index()].is_some_and(|rs| rs.owner == pid && !rs.declined),
                "peuple actif absent ou à un autre joueur",
            )?;
        }
        for (k, &d) in p.declined.iter().enumerate() {
            ok(
                s.races[d.index()].is_some_and(|rs| rs.owner == pid && rs.declined),
                "peuple en déclin absent ou à un autre joueur",
            )?;
            ok(!p.declined[..k].contains(&d) && p.active != Some(d), "peuple en déclin en double")?;
        }
    }
    for task in &s.defense {
        ok(s.races[task.race.index()].is_some_and(|rs| rs.owner == task.player), "défense d'un peuple absent")?;
    }
    ok(s.peace.len() <= s.players.len(), "trop de paix imposées")?;
    Ok(())
}

pub fn encode_state(s: &State, n: usize, w: &mut Writer) {
    w.word(s.players.len() as u32)
        .word(s.current as u32)
        .word(s.phase.index() as u32)
        .int(s.round)
        .int(s.max_rounds)
        .word(s.seq)
        .word(s.rng)
        .opt(s.last_region)
        .word(s.last_die.is_some() as u32);
    if let Some(d) = s.last_die {
        w.word(d.seq).word(d.player as u32).int(d.value).int(d.need);
        w.int(d.success.map_or(-1, i32::from)).opt(d.region).word(d.berserk as u32);
    }
    for p in &s.players {
        w.word(p.level.index() as u32).int(p.coins).race(p.active).int(p.last_income);
        w.word(p.declined.len() as u32);
        for &r in &p.declined {
            w.race(Some(r));
        }
    }
    for rs in &s.races {
        match rs {
            Some(rs) => w
                .word(rs.power.index() as u32)
                .word(rs.owner as u32)
                .word(rs.declined as u32)
                .int(rs.hand)
                .int(rs.holes_placed)
                .int(rs.picked_round),
            None => w.word(NONE_ID).word(0).word(0).int(0).int(0).int(0),
        };
    }
    for r in &s.regions[..n] {
        let occupant = match r.occupant {
            Occupant::Empty => NONE_ID,
            Occupant::Lost => LOST_ID,
            Occupant::Race(race) => race.index() as u32,
        };
        let flags = r.lair as u32
            | (r.fortress as u32) << 1
            | (r.hero as u32) << 2
            | (r.dragon as u32) << 3
            | (r.hole as u32) << 4;
        w.word(occupant).int(r.tokens).word(flags).int(r.camps);
    }
    w.word(s.market.len() as u32);
    for c in &s.market {
        w.word(c.race.index() as u32).word(c.power.index() as u32).int(c.coins);
    }
    w.word(s.race_deck.len() as u32);
    for &r in &s.race_deck {
        w.word(r.index() as u32);
    }
    w.word(s.power_deck.len() as u32);
    for &p in &s.power_deck {
        w.word(p.index() as u32);
    }
    let t = &s.turn;
    w.word((t.stage == Stage::Ghouls) as u32)
        .race(t.army)
        .int(t.conquests)
        .int(t.non_empty)
        .word(t.attacked.0 as u32)
        .word(t.sorcery.0 as u32)
        .word(t.dragon_used as u32)
        .opt(t.fortress_region)
        .int(t.berserk_die.unwrap_or(-1))
        .word(t.die_rolled as u32)
        .word(t.declined as u32)
        .word(t.picked as u32)
        .int(t.amazon_bonus);
    for &r in &t.readied[..n] {
        w.int(r);
    }
    w.word(s.defense.len() as u32);
    for d in &s.defense {
        w.word(d.player as u32).word(d.race.index() as u32);
    }
    w.word(s.peace.len() as u32);
    for p in &s.peace {
        w.word(p.diplomat as u32).word(p.target as u32);
    }
}

// --- cartes générées -------------------------------------------------------------------------

/// Une carte générée : tours, nombre de régions, puis par région : terrain, drapeaux (caverne,
/// magie, mine, tribu oubliée, bord, entrée, côte), centre (x, y), nom (0 tiré : nom commun et
/// complément ; 1 numéroté : numéro et 0), voisines, contour (nombre de points, puis x et y en f64
/// sur deux mots chacun).
pub fn encode_generated_map(m: &crate::mapgen::GeneratedMap, w: &mut Writer) {
    use crate::mapgen::Name;
    w.int(m.rounds).word(m.regions.len() as u32);
    for r in &m.regions {
        let flags = r.cavern as u32
            | (r.magic as u32) << 1
            | (r.mine as u32) << 2
            | (r.lost_tribe as u32) << 3
            | (r.border as u32) << 4
            | (r.entry as u32) << 5
            | (r.coastal as u32) << 6;
        w.word(r.terrain.index() as u32).word(flags).int(r.cx).int(r.cy);
        match r.name {
            Name::Drawn { noun, suffix } => w.word(0).word(noun).word(suffix),
            Name::Numbered(n) => w.word(1).word(n).word(0),
        };
        w.word(r.adj.len() as u32);
        for &a in &r.adj {
            w.word(a as u32);
        }
        w.word(r.outline.len() as u32);
        for &(x, y) in &r.outline {
            for v in [x, y] {
                let bits = v.to_bits();
                w.word(bits as u32).word((bits >> 32) as u32);
            }
        }
    }
}

// --- actions -----------------------------------------------------------------------------------

/// Une nouvelle partie : nombre de joueurs, niveaux, graine de la carte (2 mots), graine de la partie.
pub fn decode_new_game(r: &mut Reader) -> Result<(Vec<Level>, u64, u32)> {
    let count = r.upto(MAX_PLAYERS)?;
    if count < 2 {
        return Err(BAD);
    }
    let levels = (0..count).map(|_| Level::from_index(r.word()? as usize).ok_or(BAD)).collect::<Result<Vec<_>>>()?;
    let map_seed = r.word()? as u64 | (r.word()? as u64) << 32;
    Ok((levels, map_seed, r.word()?))
}

/// Une action : type, deux arguments, puis pour `Place` un nombre de jetons par région.
pub fn decode_action(words: &[u32], n: usize) -> Result<Action> {
    let mut r = Reader::new(words);
    let kind = r.word()?;
    let a = r.int()?;
    let b = r.int()?;
    // une région ou un indice hors limites est refusé par les règles, pas par le protocole
    let at = if a < 0 { usize::MAX } else { a as usize };
    let action = match kind {
        0 => Action::Pick { index: at },
        1 => Action::Conquer { region: at },
        2 => Action::Reinforce { region: at },
        3 => Action::BerserkRoll,
        4 => Action::Dragon { region: at },
        5 => Action::Sorcery { region: at },
        6 => Action::Abandon { region: at },
        7 => Action::Decline,
        8 => Action::EndConquest,
        9 => Action::Deploy { region: at, delta: b },
        10 => Action::AutoDeploy { all: a != 0 },
        11 => {
            let mut p = Placement::default();
            for tokens in p.0.iter_mut().take(n) {
                *tokens = r.int()?;
            }
            Action::Place(p)
        }
        12 => Action::Camp { region: at, delta: b },
        13 => Action::Fortress { region: at },
        14 => Action::Hero { region: at },
        15 => Action::EndTurn { peace: (a >= 0).then_some(at), stout: b != 0 },
        16 => Action::EndDefense,
        17 => Action::SkipPick,
        _ => return Err(BAD),
    };
    r.done()?;
    Ok(action)
}

pub fn encode_action(a: &Action, n: usize, w: &mut Writer) {
    let (kind, x, y) = match *a {
        Action::Pick { index } => (0, index as i32, 0),
        Action::Conquer { region } => (1, region as i32, 0),
        Action::Reinforce { region } => (2, region as i32, 0),
        Action::BerserkRoll => (3, 0, 0),
        Action::Dragon { region } => (4, region as i32, 0),
        Action::Sorcery { region } => (5, region as i32, 0),
        Action::Abandon { region } => (6, region as i32, 0),
        Action::Decline => (7, 0, 0),
        Action::EndConquest => (8, 0, 0),
        Action::Deploy { region, delta } => (9, region as i32, delta),
        Action::AutoDeploy { all } => (10, all as i32, 0),
        Action::Place(_) => (11, 0, 0),
        Action::Camp { region, delta } => (12, region as i32, delta),
        Action::Fortress { region } => (13, region as i32, 0),
        Action::Hero { region } => (14, region as i32, 0),
        Action::EndTurn { peace, stout } => (15, peace.map_or(-1, |p| p as i32), stout as i32),
        Action::EndDefense => (16, 0, 0),
        Action::SkipPick => (17, 0, 0),
    };
    w.word(kind).int(x).int(y);
    if let Action::Place(p) = a {
        for &tokens in &p.0[..n] {
            w.int(tokens);
        }
    }
}

// --- réponses ----------------------------------------------------------------------------------

/// Une raison de refus : code et paramètre (voir `Reason::code`).
pub fn encode_reason(reason: Option<Reason>, w: &mut Writer) {
    let (code, arg) = reason.map_or((0, 0), Reason::code);
    w.word(code).int(arg);
}

fn encode_special(s: Special, w: &mut Writer) {
    match s {
        Special::NotApplicable => w.word(0).word(0).int(0),
        Special::Allowed => w.word(1).word(0).int(0),
        Special::Denied(reason) => {
            w.word(2);
            encode_reason(Some(reason), w);
            w
        }
    };
}

fn modifier_bit(m: Modifier) -> u32 {
    1 << match m {
        Modifier::Commando => 0,
        Modifier::Mounted => 1,
        Modifier::Underworld => 2,
        Modifier::Tritons => 3,
        Modifier::Giants => 4,
    }
}

/// Par région : raison du blocage (2 mots), coût, coût effectif, type (0 aucun, 1 normal, 2 dé),
/// jetons manquants, dragon puis sorcellerie (0 sans objet, 1 permis, 2 refusé, et la raison), et
/// les modificateurs du peuple qui conquiert (bits : Commandos, Montés, Souterrains, Tritons, Géants).
pub fn encode_targets(g: &Game, w: &mut Writer) {
    let army = g.state.turn.army.filter(|_| g.state.phase == Phase::Conquest);
    for t in g.targets() {
        let mods = army.map_or(0, |a| g.attack_modifiers(a, t.region).iter().map(|&m| modifier_bit(m)).sum());
        encode_reason(t.block, w);
        let kind = match t.kind {
            TargetKind::None => 0,
            TargetKind::Normal => 1,
            TargetKind::Die => 2,
        };
        w.int(t.cost).int(t.effective).word(kind).int(t.need);
        encode_special(t.dragon, w);
        encode_special(t.sorcery, w);
        w.word(mods);
    }
}

/// Les attaquants d'une région : joueur, peuple (255 : un peuple à choisir), coût, manière (0
/// conquête, 1 dragon, 2 sorcellerie), aide (0 aucune, 1 modificateur, 2 Berserks) et modificateur,
/// accès lointain (0 voisin, 1 Volants, 2 cavernes).
pub fn encode_attackers(attackers: &[Attacker], w: &mut Writer) {
    for a in attackers {
        let how = match a.how {
            AttackHow::Conquer => 0,
            AttackHow::Dragon => 1,
            AttackHow::Sorcery => 2,
        };
        let (hint, modifier) = match a.hint {
            None => (0, 0),
            Some(AttackHint::Modifier(m)) => (1, modifier_bit(m)),
            Some(AttackHint::Berserk) => (2, 0),
        };
        let via = match a.via {
            None => 0,
            Some(Via::Flying) => 1,
            Some(Via::Cavern) => 2,
        };
        w.word(a.player as u32).race(a.race).int(a.cost).word(how).word(hint).word(modifier).word(via);
    }
}

/// Les lignes d'un décompte : type, peuple, en déclin, valeur.
pub fn encode_income(lines: &[IncomeLine], w: &mut Writer) {
    for l in lines {
        w.word(income_kind(l.kind)).word(l.race.index() as u32).word(l.declined as u32).int(l.value);
    }
}

fn income_kind(k: IncomeKind) -> u32 {
    k as u32
}

/// Les événements, chacun précédé de sa longueur : longueur, numéro (seq), manche, type, champs.
pub fn encode_events(events: &[Logged], w: &mut Writer) {
    for &Logged { seq, round, ref event } in events {
        let e = event;
        let start = w.len();
        w.word(0).word(seq).int(round);
        let race = |r: Race| r.index() as u32;
        let power = |p: Power| p.index() as u32;
        match *e {
            Event::TurnStarted { player, race: r, power: p, hand, ghouls, amazon_bonus } => {
                w.word(0)
                    .word(player as u32)
                    .word(race(r))
                    .word(power(p))
                    .int(hand)
                    .word(ghouls as u32)
                    .int(amazon_bonus);
            }
            Event::Picked { player, race: r, power: p, tokens, paid, gained } => {
                w.word(1).word(player as u32).word(race(r)).word(power(p)).int(tokens).int(paid).int(gained);
            }
            Event::Conquered {
                player,
                race: r,
                region,
                how,
                tokens,
                victim,
                victim_occupant,
                victim_active,
                lost,
                returned,
            } => {
                let how = match how {
                    How::Conquer => 0,
                    How::Die => 1,
                    How::Dragon => 2,
                    How::Sorcery => 3,
                };
                let occupant = match victim_occupant {
                    Occupant::Empty => NONE_ID,
                    Occupant::Lost => LOST_ID,
                    Occupant::Race(v) => race(v),
                };
                w.word(2).word(player as u32).word(race(r)).word(region as u32).word(how).int(tokens);
                w.opt(victim).word(occupant).word(victim_active as u32).int(lost).int(returned);
            }
            Event::DieRolled { player, region, value, need, success } => {
                w.word(3).word(player as u32).opt(region).int(value).int(need).int(success.map_or(-1, i32::from));
            }
            Event::Abandoned { player, region } => {
                w.word(4).word(player as u32).word(region as u32);
            }
            Event::Declined { player, race: r, power: p, removed } => {
                w.word(5).word(player as u32).word(race(r)).word(power(p)).word(removed.len() as u32);
                for &x in &removed {
                    w.word(race(x));
                }
            }
            Event::Scored { player, gain, lines } => {
                w.word(6).word(player as u32).int(gain).word(lines.len() as u32);
                encode_income(&lines, w);
            }
            Event::PeaceImposed { player, target, race: r } => {
                w.word(7).word(player as u32).word(target as u32).race(r);
            }
            Event::AmazonsReturned { player, tokens } => {
                w.word(8).word(player as u32).int(tokens);
            }
            Event::SkeletonsRaised { player, tokens } => {
                w.word(9).word(player as u32).int(tokens);
            }
            Event::GhoulsDone { player } => {
                w.word(10).word(player as u32);
            }
            Event::Defended { player, race: r } => {
                w.word(11).word(player as u32).word(race(r));
            }
            Event::GameOver => {
                w.word(12);
            }
            Event::GameStarted => {
                w.word(13);
            }
        }
        let len = w.len() - start;
        w.out[start] = len as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::NoJournal;
    use crate::mapgen;

    #[test]
    fn games_round_trip() {
        let generated = mapgen::generate(3, 77).unwrap();
        let map = generated.to_map().unwrap();
        let lost = generated.regions.iter().enumerate().filter(|(_, r)| r.lost_tribe).map(|(i, _)| i).collect();
        let g = Game::create(&map, lost, &[Level::Normal; 3], generated.rounds, 5, &mut NoJournal);
        let mut buf = vec![0; 8192];
        let mut w = Writer::new(&mut buf);
        encode_game(&map, &g.state, &mut w);
        let (map2, state2) = decode_game(w.written()).unwrap();
        assert_eq!(state2, g.state);
        assert_eq!(map2.regions, map.regions);
    }
}
