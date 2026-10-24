//! Ce qu'un joueur sait vraiment, et les mondes possibles qui vont avec.
//!
//! L'état du noyau contient tout, y compris ce qu'aucun joueur ne voit :
//! - l'ordre des pioches (peuples et pouvoirs). Leur contenu est connu (tout ce qui n'est ni en jeu
//!   ni au marché) ; seul l'ordre est caché, sauf pour les cartes rendues sous la pioche quand un
//!   peuple disparaît : leur ordre est public, et un joueur attentif s'en souvient ;
//! - l'état du générateur des dés, qui donne les dés à venir.
//!
//! Les pièces des autres, cachées dans l'interface, se recalculent exactement à partir de ce qui est
//! annoncé (5 au départ, prix et pièces de chaque choix de peuple, gain de chaque tour) : un joueur
//! qui compte les connaît. L'IA a donc droit aux pièces de l'état, comme un joueur humain qui compte.
//!
//! Une IA honnête ne regarde l'état qu'à travers `determinize`, qui remplace ce qui est caché par un
//! tirage compatible avec ce que le joueur sait.

use smallmobile_kernel::action::{Event, Journal, Logged};
use smallmobile_kernel::ids::MAX_PLAYERS;
use smallmobile_kernel::state::Occupant;
use smallmobile_kernel::{Game, Phase, State};

use crate::moves::{Kind, Move, Redeploy};
use crate::rand::Rng;

/// Ce qu'un joueur attentif sait : le nombre de cartes connues au fond de chaque pioche, et ce que
/// chaque joueur a fait depuis le début de la partie (`History`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Knowledge {
    pub race_tail: usize,
    pub power_tail: usize,
    pub history: History,
    /// Où en est la fin de tour en cours, découpée en coups (`moves::Redeploy`) : rien d'un
    /// redéploiement joué en actions du noyau.
    pub redeploy: Redeploy,
    /// Les jetons redéployés dans l'ordre des régions (`moves`, `search::Config::ordered_tokens`) :
    /// un réglage de la recherche, pas un savoir sur la partie.
    pub ordered_tokens: bool,
}

/// Les habitudes d'un joueur (`History::habits`) : des comptes depuis le début de la partie.
pub mod habit {
    /// Régions vides prises.
    pub const EMPTY: usize = 0;
    /// Régions prises aux tribus oubliées.
    pub const LOST_TRIBES: usize = 1;
    /// Régions prises au peuple actif d'un adversaire.
    pub const ACTIVE: usize = 2;
    /// Régions prises au peuple en déclin d'un adversaire.
    pub const DECLINED: usize = 3;
    /// Régions prises à l'adversaire le plus riche du moment.
    pub const LEADER: usize = 4;
    /// Déclins, et manches tenues par les peuples déclinés (somme).
    pub const DECLINES: usize = 5;
    pub const DECLINE_AGE: usize = 6;
    /// Peuples choisis, pièces payées (place au marché) et pièces ramassées avec eux (sommes).
    pub const PICKS: usize = 7;
    pub const PAID: usize = 8;
    pub const GAINED: usize = 9;
    /// Paix imposées (Diplomates).
    pub const PEACE: usize = 10;
    /// Régions abandonnées.
    pub const ABANDONS: usize = 11;
    /// Dés de renfort lancés.
    pub const DICE: usize = 12;
    /// Jetons perdus sous les attaques des autres.
    pub const TOKENS_LOST: usize = 13;
    pub const COUNT: usize = 14;
}

/// Ce que chaque joueur a fait depuis le début de la partie, d'après ses coups publics : de quoi
/// deviner son style, comme un joueur attentif (attaque-t-il, et qui ? décline-t-il tôt ? paie-t-il
/// cher ses peuples ?). Joueurs réels ; `encode` les range par joueur relatif. Rien de caché : la
/// même histoire pour tous les joueurs, refaite à l'identique en rejouant les actions de la partie.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct History {
    pub habits: [[u16; habit::COUNT]; MAX_PLAYERS],
    /// Régions prises par chaque joueur à chaque joueur (peuple actif ou en déclin, les siens compris).
    pub attacks: [[u16; MAX_PLAYERS]; MAX_PLAYERS],
    /// De même, pendant le dernier tour de chaque joueur (celui en cours compris).
    pub recent: [[u8; MAX_PLAYERS]; MAX_PLAYERS],
    /// Le joueur dont c'est le tour (aucun avant le premier).
    turn: Option<u8>,
    /// La manche où chaque joueur a choisi son dernier peuple.
    picked: [u8; MAX_PLAYERS],
}

fn add(x: &mut u16, v: i32) {
    *x = x.saturating_add(v.clamp(0, u16::MAX as i32) as u16);
}

impl History {
    /// Le tour de `p` commence (son premier événement : choix du peuple ou début des conquêtes).
    fn start(&mut self, p: usize) {
        if self.turn != Some(p as u8) {
            self.turn = Some(p as u8);
            self.recent[p] = [0; MAX_PLAYERS];
        }
    }
}

/// Le journal d'un coup (`Knowledge::journal`), pour `Knowledge::observe` : les cartes rendues
/// sous les pioches, et l'histoire des joueurs, mise à jour événement par événement (sans allouer).
#[derive(Default)]
pub struct Returns {
    pub returned: usize,
    history: History,
    /// Les pièces de chaque joueur, suivies d'un événement à l'autre (pour `habit::LEADER`).
    coins: [i32; MAX_PLAYERS],
    players: usize,
}

impl Journal for Returns {
    fn record(&mut self, entry: Logged) {
        use habit::*;
        let h = &mut self.history;
        match entry.event {
            Event::Picked { player, paid, gained, .. } => {
                h.start(player);
                let x = &mut h.habits[player];
                add(&mut x[PICKS], 1);
                add(&mut x[PAID], paid);
                add(&mut x[GAINED], gained);
                h.picked[player] = entry.round.clamp(0, 255) as u8;
                self.coins[player] += gained - paid;
            }
            Event::TurnStarted { player, .. } => h.start(player),
            Event::Conquered { player, victim, victim_occupant, victim_active, lost, .. } => {
                let x = &mut h.habits[player];
                match (victim, victim_occupant) {
                    (Some(v), _) => {
                        add(&mut h.attacks[player][v], 1);
                        h.recent[player][v] = h.recent[player][v].saturating_add(1);
                        if v != player {
                            add(&mut x[if victim_active { ACTIVE } else { DECLINED }], 1);
                            let richest = (0..self.players).filter(|&q| q != player).map(|q| self.coins[q]).max();
                            if richest == Some(self.coins[v]) {
                                add(&mut x[LEADER], 1);
                            }
                            add(&mut h.habits[v][TOKENS_LOST], lost);
                        }
                    }
                    (None, Occupant::Lost) => add(&mut x[LOST_TRIBES], 1),
                    (None, _) => add(&mut x[EMPTY], 1),
                }
            }
            Event::DieRolled { player, region: Some(_), .. } => add(&mut h.habits[player][DICE], 1),
            Event::Abandoned { player, .. } => add(&mut h.habits[player][ABANDONS], 1),
            Event::Declined { player, removed, .. } => {
                self.returned += removed.len();
                add(&mut h.habits[player][DECLINES], 1);
                add(&mut h.habits[player][DECLINE_AGE], entry.round - h.picked[player] as i32);
            }
            Event::PeaceImposed { player, .. } => add(&mut h.habits[player][PEACE], 1),
            Event::Scored { player, gain, .. } => self.coins[player] += gain,
            _ => {}
        }
    }
}

impl Knowledge {
    /// Ce que sait un joueur de la partie en cours : le noyau suit le fond public des pioches.
    /// L'histoire des joueurs reste vide : elle se refait en rejouant la partie depuis le début.
    pub fn of(s: &State) -> Knowledge {
        Knowledge { race_tail: s.race_tail as usize, power_tail: s.power_tail as usize, ..Knowledge::default() }
    }

    /// Le journal à passer au noyau pour l'action ou le coup qui suit, joué depuis `before`.
    pub fn journal(&self, before: &State) -> Returns {
        let mut coins = [0; MAX_PLAYERS];
        for (c, p) in coins.iter_mut().zip(&before.players) {
            *c = p.coins;
        }
        Returns { returned: 0, history: self.history, coins, players: before.player_count() }
    }

    /// Met à jour ce qui est su après une action ou un coup : son journal (`journal`), puis les
    /// pioches de l'état qui en résulte (les cartes tirées partent du dessus).
    pub fn observe(&mut self, ret: Returns, after: &State) {
        self.race_tail = (self.race_tail + ret.returned).min(after.race_deck.len());
        self.power_tail = (self.power_tail + ret.returned).min(after.power_deck.len());
        self.history = ret.history;
        if after.phase != Phase::Redeploy {
            self.redeploy = Redeploy::default();
        }
    }

    /// Joue une action du noyau en suivant ce qui est su.
    pub fn apply(&mut self, g: &mut Game, a: &smallmobile_kernel::Action) -> Result<(), smallmobile_kernel::Reason> {
        let mut ret = self.journal(&g.state);
        g.apply(a, &mut ret)?;
        self.observe(ret, &g.state);
        Ok(())
    }

    /// Joue un coup en suivant ce qui est su.
    pub fn play(&mut self, g: &mut Game, mv: &Move) -> Result<(), smallmobile_kernel::Reason> {
        let mut ret = self.journal(&g.state);
        crate::moves::play(g, mv, &mut ret)?;
        self.observe(ret, &g.state);
        if g.state.phase == Phase::Redeploy {
            match mv {
                Move::Lift => self.redeploy.lifted = true,
                Move::Region(Kind::Token, r) => {
                    self.redeploy.powers_done = true;
                    self.redeploy.last_token = *r;
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Un monde possible : même partie vue par les joueurs, l'ordre caché des pioches et les dés à
/// venir tirés au hasard. Le monde ne dépend que de ce que sait le joueur et de `rng` : la partie
/// cachée des pioches est remise dans un ordre fixe avant d'être mélangée.
pub fn determinize<'m>(g: &Game<'m>, k: &Knowledge, rng: &mut Rng) -> Game<'m> {
    let mut w = *g;
    let s = &mut w.state;
    let hidden = s.race_deck.len() - k.race_tail.min(s.race_deck.len());
    let races = &mut s.race_deck.as_mut_slice()[..hidden];
    races.sort_unstable();
    rng.shuffle(races);
    let hidden = s.power_deck.len() - k.power_tail.min(s.power_deck.len());
    let powers = &mut s.power_deck.as_mut_slice()[..hidden];
    powers.sort_unstable();
    rng.shuffle(powers);
    reseed_dice(&mut w, rng);
    w
}

/// Les dés à venir redeviennent imprévisibles (sans toucher aux pioches).
pub fn reseed_dice(g: &mut Game, rng: &mut Rng) {
    g.state.rng = rng.next_u32();
}
