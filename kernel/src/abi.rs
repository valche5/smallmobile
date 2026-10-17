//! Les fonctions exportées par le module WebAssembly du jeu.
//!
//! Le noyau garde une partie en mémoire. Les données passent par un tampon de mots de 32 bits
//! (`kernel_buffer`) : l'appelant y écrit sa demande, appelle la fonction avec le nombre de mots,
//! puis lit la réponse dans le même tampon. Une fonction renvoie le nombre de mots de sa réponse, ou
//! un code négatif : `MALFORMED`, `NO_GAME`, ou `REFUSED` (la raison est alors dans le tampon : code
//! puis paramètre).

use std::cell::RefCell;

use crate::action::Logged;
use crate::ai::{Ai, Restrictions, Tuning};
use crate::game::Game;
use crate::ids::Race;
use crate::map::Map;
use crate::mapgen;
use crate::reason::Reason;
use crate::state::{Occupant, State};
use crate::wire::{self, Malformed, Writer};

/// Version du protocole : à changer à chaque modification incompatible.
pub const ABI_VERSION: u32 = 2;
pub const CAPACITY: usize = 16 * 1024;

pub const MALFORMED: i32 = -1;
pub const NO_GAME: i32 = -2;
pub const REFUSED: i32 = -3;

struct Kernel {
    buffer: Vec<u32>,
    game: Option<(Map, State)>,
}

thread_local! {
    static KERNEL: RefCell<Kernel> =
        RefCell::new(Kernel { buffer: vec![0; CAPACITY], game: None });
}

enum Failure {
    Malformed(&'static str),
    NoGame,
    Refused(Reason),
}

impl From<Malformed> for Failure {
    fn from(m: Malformed) -> Self {
        Failure::Malformed(m.0)
    }
}

impl From<Reason> for Failure {
    fn from(r: Reason) -> Self {
        Failure::Refused(r)
    }
}

/// Exécute `f` sur le noyau, avec les `len` premiers mots du tampon comme demande.
fn call(len: u32, f: impl FnOnce(&mut Kernel, &[u32], &mut Writer) -> Result<(), Failure>) -> i32 {
    KERNEL.with_borrow_mut(|k| {
        let len = len as usize;
        if len > CAPACITY {
            return MALFORMED;
        }
        let input: Vec<u32> = k.buffer[..len].to_vec();
        let mut out = std::mem::take(&mut k.buffer);
        let result = {
            let mut w = Writer::new(&mut out);
            f(k, &input, &mut w).map(|()| w.len())
        };
        let code = match result {
            Ok(n) => n as i32,
            Err(Failure::Malformed(why)) => {
                // la raison, en UTF-8 : nombre d'octets, puis les octets rangés dans des mots
                let mut w = Writer::new(&mut out);
                w.word(why.len() as u32);
                for chunk in why.as_bytes().chunks(4) {
                    let mut b = [0u8; 4];
                    b[..chunk.len()].copy_from_slice(chunk);
                    w.word(u32::from_le_bytes(b));
                }
                MALFORMED
            }
            Err(Failure::NoGame) => NO_GAME,
            Err(Failure::Refused(reason)) => {
                wire::encode_reason(Some(reason), &mut Writer::new(&mut out));
                REFUSED
            }
        };
        k.buffer = out;
        code
    })
}

impl Kernel {
    fn game(&mut self) -> Result<(&Map, &mut State), Failure> {
        let (map, state) = self.game.as_mut().ok_or(Failure::NoGame)?;
        Ok((map, state))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn kernel_version() -> u32 {
    ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn kernel_capacity() -> u32 {
    CAPACITY as u32
}

/// Adresse du tampon d'échange (fixe pour toute la vie du noyau).
#[unsafe(no_mangle)]
pub extern "C" fn kernel_buffer() -> *mut u32 {
    KERNEL.with_borrow_mut(|k| k.buffer.as_mut_ptr())
}

/// Charge une partie (carte et état, voir `wire::decode_game`).
#[unsafe(no_mangle)]
pub extern "C" fn kernel_load(len: u32) -> i32 {
    call(len, |k, input, _| {
        k.game = Some(wire::decode_game(input)?);
        Ok(())
    })
}

/// Remplace l'état de la partie chargée (même carte).
#[unsafe(no_mangle)]
pub extern "C" fn kernel_set_state(len: u32) -> i32 {
    call(len, |k, input, _| {
        let (map, state) = k.game()?;
        let mut r = wire::Reader::new(input);
        *state = wire::decode_state(&mut r, map)?;
        r.done()?;
        Ok(())
    })
}

/// La carte générée pour `players` joueurs et cette graine (voir `wire::encode_generated_map`).
#[unsafe(no_mangle)]
pub extern "C" fn kernel_generate_map(players: u32, seed_lo: u32, seed_hi: u32) -> i32 {
    call(0, |_, _, w| {
        let seed = seed_lo as u64 | (seed_hi as u64) << 32;
        let map = mapgen::generate(players as usize, seed).map_err(|_| Failure::Malformed("carte impossible"))?;
        wire::encode_generated_map(&map, w);
        Ok(())
    })
}

/// Crée et charge une nouvelle partie : nombre de joueurs, leurs niveaux, graine de la carte (2 mots),
/// graine de la partie. Renvoie les événements du début de partie.
#[unsafe(no_mangle)]
pub extern "C" fn kernel_new_game(len: u32) -> i32 {
    call(len, |k, input, w| {
        let mut r = wire::Reader::new(input);
        let (levels, map_seed, seed) = wire::decode_new_game(&mut r)?;
        r.done()?;
        let generated = mapgen::generate(levels.len(), map_seed).map_err(|_| Failure::Malformed("carte impossible"))?;
        let map = generated.to_map().map_err(|_| Failure::Malformed("carte impossible"))?;
        let lost = generated.regions.iter().enumerate().filter(|(_, r)| r.lost_tribe).map(|(i, _)| i).collect();
        let mut events: Vec<Logged> = Vec::new();
        let state = Game::create(&map, lost, &levels, generated.rounds, seed, &mut events).state;
        k.game = Some((map, state));
        wire::encode_events(&events, w);
        Ok(())
    })
}

/// L'état de la partie chargée.
#[unsafe(no_mangle)]
pub extern "C" fn kernel_state() -> i32 {
    call(0, |k, _, w| {
        let (map, state) = k.game()?;
        wire::encode_state(state, map.len(), w);
        Ok(())
    })
}

/// Joue une action ; renvoie les événements produits.
#[unsafe(no_mangle)]
pub extern "C" fn kernel_apply(len: u32) -> i32 {
    call(len, |k, input, w| {
        let (map, state) = k.game()?;
        let action = wire::decode_action(input, map.len())?;
        let mut game = Game::new(map, *state);
        let mut events: Vec<Logged> = Vec::new();
        game.apply(&action, &mut events)?;
        *state = game.state;
        wire::encode_events(&events, w);
        Ok(())
    })
}

/// Commence la partie chargée (marché, premier tour) ; renvoie les événements produits.
#[unsafe(no_mangle)]
pub extern "C" fn kernel_start() -> i32 {
    call(0, |k, _, w| {
        let (map, state) = k.game()?;
        let mut game = Game::new(map, *state);
        let mut events: Vec<Logged> = Vec::new();
        game.start(&mut events);
        *state = game.state;
        wire::encode_events(&events, w);
        Ok(())
    })
}

/// La décision de l'IA pour le joueur qui doit agir (bit 0 : sans déclin, bit 1 : sans replacer ses jetons).
#[unsafe(no_mangle)]
pub extern "C" fn kernel_decide(flags: u32) -> i32 {
    call(0, |k, _, w| {
        let tuning = Tuning::default();
        let (map, state) = k.game()?;
        let restrictions = Restrictions { no_decline: flags & 1 != 0, keep_placement: flags & 2 != 0 };
        let action = Ai::new(&tuning).decide(&Game::new(map, *state), restrictions)?;
        wire::encode_action(&action, map.len(), w);
        Ok(())
    })
}

/// Les régions vues par le peuple qui conquiert (voir `wire::encode_targets`).
#[unsafe(no_mangle)]
pub extern "C" fn kernel_targets() -> i32 {
    call(0, |k, _, w| {
        let (map, state) = k.game()?;
        wire::encode_targets(&Game::new(map, *state), w);
        Ok(())
    })
}

/// Les adversaires qui pourraient prendre `region` à `owner` avant son prochain tour.
#[unsafe(no_mangle)]
pub extern "C" fn kernel_attackers(region: u32, owner: u32) -> i32 {
    query(|g, w| {
        let (region, owner) = (region as usize, owner as usize);
        if region >= g.map.len() || owner >= g.state.player_count() {
            return Err(Failure::Malformed("demande hors limites"));
        }
        wire::encode_attackers(&g.attackers(region, owner), w);
        Ok(())
    })
}

/// Le décompte que rapporterait la fin du tour pour ce joueur, maintenant.
#[unsafe(no_mangle)]
pub extern "C" fn kernel_income(player: u32) -> i32 {
    query(|g, w| {
        if player as usize >= g.state.player_count() {
            return Err(Failure::Malformed("demande hors limites"));
        }
        wire::encode_income(&g.income_lines(player as usize), w);
        Ok(())
    })
}

/// Ce qui est permis maintenant : raison de refuser le déclin (0 : permis), puis pour chaque région
/// la raison de refuser son abandon (chaque raison : code et paramètre).
#[unsafe(no_mangle)]
pub extern "C" fn kernel_permissions() -> i32 {
    query(|g, w| {
        wire::encode_reason(g.can_decline().err(), w);
        for region in 0..g.map.len() {
            wire::encode_reason(g.can_abandon(region).err(), w);
        }
        Ok(())
    })
}

/// La menace qui pèse sur une région du peuple (f64 sur deux mots).
#[unsafe(no_mangle)]
pub extern "C" fn kernel_threat(race: u32, region: u32) -> i32 {
    query(|g, w| {
        let race = in_play(g, race)?;
        if region as usize >= g.map.len() {
            return Err(Failure::Malformed("demande hors limites"));
        }
        let bits = g.threat(race, region as usize).to_bits();
        w.word(bits as u32).word((bits >> 32) as u32);
        Ok(())
    })
}

/// Le placement complet que ferait « tout replacer » : jetons visés par région.
#[unsafe(no_mangle)]
pub extern "C" fn kernel_full_deployment(race: u32) -> i32 {
    query(|g, w| {
        let race = in_play(g, race)?;
        let target = g.full_deployment(race);
        for r in 0..g.map.len() {
            w.int(if g.state.regions[r].occupant == Occupant::Race(race) { target.0[r] } else { 0 });
        }
        Ok(())
    })
}

/// L'état juste après un passage en déclin du joueur en cours (avant le décompte), pour les aperçus.
#[unsafe(no_mangle)]
pub extern "C" fn kernel_decline_preview() -> i32 {
    query(|g, w| {
        let active = g.state.players[g.state.current].active;
        if active.is_none() {
            return Err(Failure::Refused(Reason::NoActiveRace));
        }
        wire::encode_state(&g.declined_preview().state, g.map.len(), w);
        Ok(())
    })
}

/// Un peuple en jeu, d'après son numéro.
fn in_play(g: &Game, race: u32) -> Result<Race, Failure> {
    Race::from_index(race as usize)
        .filter(|r| g.state.races[r.index()].is_some())
        .ok_or(Failure::Malformed("demande hors limites"))
}

/// Une question sur la partie chargée, sans la modifier.
fn query(f: impl FnOnce(&Game, &mut Writer) -> Result<(), Failure>) -> i32 {
    call(0, |k, _, w| {
        let (map, state) = k.game()?;
        f(&Game::new(map, *state), w)
    })
}
