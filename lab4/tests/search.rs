//! Tests de positions et de recherche ; aucune arène ni campagne d'entraînement.
use std::sync::Arc;

use smallmobile_kernel::Phase;
use smallmobile_lab4::agent::{Agent, KernelAi};
use smallmobile_lab4::az::{Az, Candidates, Config, Evals};
use smallmobile_lab4::know::Knowledge;
use smallmobile_lab4::moves;
use smallmobile_lab4::net::Net;
use smallmobile_lab4::rand::Rng;
use smallmobile_lab4::table::Table;

/// Même réseau que Direct, avec la limite d'évaluations simultanées réservée à une partie GPU.
/// Détecte une recherche élargie qui enverrait tous ses candidats d'un coup dans l'anneau.
#[derive(Default)]
struct LimitedOracle {
    pending: std::cell::Cell<usize>,
    peak: std::cell::Cell<usize>,
}

impl smallmobile_lab4::az::Oracle for LimitedOracle {
    fn eval<'s>(
        &'s self,
        net: &Net,
        e: &smallmobile_lab4::encode::Encoded,
        map: &smallmobile_kernel::Map,
    ) -> impl Future<Output = smallmobile_lab4::net::Eval> + use<'s> {
        let pending = self.pending.get() + 1;
        assert!(pending <= 16, "trop de feuilles simultanées pour une partie GPU");
        self.pending.set(pending);
        self.peak.set(self.peak.get().max(pending));
        let value = net.eval(e, map);
        async move {
            self.pending.set(self.pending.get() - 1);
            value
        }
    }
}

fn weights(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("nets").join(name)
}

#[test]
fn configurations_are_explicit_and_invalid_budgets_are_rejected() {
    let cfg = Config::parse("n=3200,candidates=all").unwrap();
    assert_eq!(cfg.iterations, 3200);
    assert_eq!(cfg.candidates, Candidates::All);
    assert_eq!(Config::default().iterations, 400);
    assert_eq!(Config::default().candidates, Candidates::Top(16));
    assert_eq!(Config::parse("candidates=32").unwrap().candidates, Candidates::Top(32));
    assert!(Config::parse("tokens=ordered").unwrap().ordered_tokens);
    assert!(!Config::default().ordered_tokens);
    for bad in ["n=0", "n=-1", "n=no", "candidates=0", "candidates=-1", "candidates=no", "width=16", "tokens=any"] {
        assert!(Config::parse(bad).is_none(), "{bad}");
    }
}

#[test]
fn reference_matches_frozen_lab3_with_c130_and_the_same_random_stream() {
    use smallmobile_brain3 as old;
    let bytes = std::fs::read(weights("c130.bin")).unwrap();
    // Le budget de production et le bruit du selfplay sont tous deux couverts.
    let new = Az::new(Config::default(), Arc::new(Net::from_bytes(&bytes).unwrap()), "c130".into());
    let old = old::search::Az::new(
        old::search::Config { iterations: 400 },
        Arc::new(old::net::Net::from_bytes(&bytes).unwrap()),
        "c130".into(),
    );
    let normal = KernelAi::new(smallmobile_kernel::Level::Normal);
    let mut checked = 0;
    for players in 2..=5 {
        let table = Table::new(players, 174 + players as u64);
        let mut g = table.game(18);
        let mut k = Knowledge::default();
        let mut ko = old::know::Knowledge::default();
        let mut rng = Rng::new(19);
        for step in 0..75 {
            if g.state.phase == Phase::GameOver {
                break;
            }
            if step % 23 == 0 && moves::legal(&g, &k).len() > 1 {
                let explore = step != 0;
                let mut rng_new = Rng::new(20 + step);
                let mut rng_old = old::rand::Rng::new(20 + step);
                let a = new.search(&g, &k, &mut rng_new, explore, &mut Evals::default());
                let b = old.search(&g, &ko, &mut rng_old, explore, &mut old::search::Evals::default());
                assert_eq!(format!("{:?}", a.mv), format!("{:?}", b.mv));
                assert_eq!(a.visits, b.visits);
                assert_eq!(a.policy, b.policy);
                assert_eq!(a.value.to_bits(), b.value.to_bits());
                assert_eq!(a.finals, b.finals);
                assert_eq!(a.wins, b.wins);
                assert_eq!(rng_new.next_u64(), rng_old.next_u64());
                assert_eq!(a.diagnostics.simulations, 400);
                checked += 1;
            }
            for action in normal.act(&g, &k, &mut rng) {
                let mut copy = g;
                ko.apply(&mut copy, &action).unwrap();
                k.apply(&mut g, &action).unwrap();
            }
        }
    }
    assert!(checked >= 8, "{checked} positions comparées");
}

#[test]
fn widening_visits_every_legal_move_and_reports_actual_budget_and_rank() {
    let net = Arc::new(Net::load(&weights("gen0.bin")).unwrap());
    let narrow = Az::new(Config { iterations: 128, candidates: Candidates::Top(16), ordered_tokens: false }, net.clone(), "zero".into());
    let wide = Az::new(Config { iterations: 128, candidates: Candidates::All, ordered_tokens: false }, net.clone(), "zero".into());
    let tiny = Az::new(Config { iterations: 7, candidates: Candidates::All, ordered_tokens: false }, net, "zero".into());
    let normal = KernelAi::new(smallmobile_kernel::Level::Normal);
    // Trouver une position réelle avec plus de 16 choix, sans jouer d'arène.
    for seed in 1..=8 {
        let table = Table::new(4, 700 + seed);
        let mut g = table.game(seed as u32);
        let mut k = Knowledge::default();
        let mut rng = Rng::new(seed);
        for _ in 0..100 {
            if g.state.phase == Phase::GameOver {
                break;
            }
            let legal = moves::legal(&g, &k);
            if legal.len() > 16 && legal.len() <= 128 {
                let search = |az: &Az| az.search(&g, &k, &mut Rng::new(80), false, &mut Evals::default());
                let n = search(&narrow);
                let w = search(&wide);
                let t = search(&tiny);
                assert_eq!(n.diagnostics.visited, 16);
                assert_eq!(n.diagnostics.visited_outside16, 0);
                assert_eq!(n.diagnostics.chosen_outside16, 0);
                assert_eq!(w.diagnostics.visited, legal.len() as u64);
                assert_eq!(w.diagnostics.visited_outside16, legal.len() as u64 - 16);
                assert!(w.visits.iter().all(|(_, visits)| *visits > 0));
                // Gen0 a les mêmes logits partout : l'ordre des égalités est celui du moteur.
                let chosen = legal.iter().position(|c| c.mv == w.mv).unwrap();
                assert_eq!(w.diagnostics.chosen_outside16, u64::from(chosen < legal.len() - 16));
                assert_eq!(t.diagnostics.visited, 7);
                for (s, budget) in [(&n, 128), (&w, 128), (&t, 7)] {
                    assert_eq!(s.visits.iter().map(|(_, n)| u64::from(*n)).sum::<u64>(), budget);
                    assert_eq!(s.diagnostics.simulations, budget);
                    assert!(s.diagnostics.depth_sum >= budget);
                    assert!(s.diagnostics.next_turn <= s.diagnostics.turn_finished);
                    assert!(s.diagnostics.turn_finished <= budget);
                    assert!(s.diagnostics.game_over <= budget);
                }
                assert_eq!(wide.diagnostics()[2], w.diagnostics);
                let oracle = LimitedOracle::default();
                let bounded = smallmobile_lab4::az::now(wide.run(
                    &g,
                    &k,
                    &mut Rng::new(80),
                    128,
                    false,
                    &mut Evals::default(),
                    &oracle,
                ));
                assert_eq!(bounded.mv, w.mv);
                assert_eq!(bounded.visits, w.visits);
                assert_eq!(bounded.policy, w.policy);
                assert_eq!(bounded.value.to_bits(), w.value.to_bits());
                assert_eq!(bounded.diagnostics, w.diagnostics);
                assert!(oracle.peak.get() > 1);
                assert_eq!(oracle.pending.get(), 0);
                let mut out = Vec::new();
                smallmobile_lab4::diagnostics::write(&mut out, 80, &[(0, &wide)]).unwrap();
                let text = String::from_utf8(out).unwrap();
                assert!(text.contains("candidates=all"));
                let lines: Vec<_> = text.lines().collect();
                assert_eq!(lines.len(), 3);
                assert_eq!(lines[1].split('\t').count(), lines[2].split('\t').count());
                return;
            }
            for action in normal.act(&g, &k, &mut rng) {
                k.apply(&mut g, &action).unwrap();
            }
        }
    }
    panic!("aucune position à plus de 16 coups dans les fixtures");
}

/// Toutes les positions de fin de pose des jetons depuis `g` (en redéploiement), une par suite de
/// coups : jetons libres ou dans l'ordre des régions.
fn token_ends<'m>(g: &smallmobile_kernel::Game<'m>, k: &Knowledge, ordered: bool, out: &mut Vec<smallmobile_kernel::state::State>) {
    let k = Knowledge { ordered_tokens: ordered, ..*k };
    let legal = moves::legal(g, &k);
    if !legal.iter().any(|c| matches!(c.mv, moves::Move::Region(moves::Kind::Token, _))) {
        out.push(g.state);
        return;
    }
    for c in &legal {
        let (mut g, mut k) = (*g, k);
        k.play(&mut g, &c.mv).unwrap();
        token_ends(&g, &k, ordered, out);
    }
}

#[test]
fn ordered_tokens_reach_the_same_redeployments_once_each() {
    let mut rng = Rng::new(5);
    let mut checked = 0;
    for deal in 0..60u64 {
        let table = Table::new(2 + (deal % 4) as usize, 1000 + deal);
        let (mut g, mut k) = (table.game(deal as u32), Knowledge::default());
        while g.state.phase != Phase::GameOver {
            let legal = moves::legal(&g, &k);
            // au premier jeton d'une pose (pouvoirs finis), assez petite pour tout énumérer
            let tokens = legal.iter().all(|c| matches!(c.mv, moves::Move::Region(moves::Kind::Token, _)));
            let army = g.state.turn.army;
            let hand = army.and_then(|a| g.state.races[a.index()]).map_or(0, |rs| rs.hand);
            if g.state.phase == Phase::Redeploy && tokens && !k.redeploy.powers_done && legal.len() > 1 && hand <= 4 && legal.len() <= 6 {
                let (mut free, mut ordered) = (Vec::new(), Vec::new());
                token_ends(&g, &k, false, &mut free);
                token_ends(&g, &k, true, &mut ordered);
                let mut distinct: Vec<_> = Vec::new();
                for s in &free {
                    if !distinct.contains(s) {
                        distinct.push(*s);
                    }
                }
                assert!(free.len() > ordered.len() || hand == 1, "moins de suites dans l'ordre");
                assert_eq!(ordered.len(), distinct.len(), "une suite par position, dans l'ordre");
                assert!(distinct.iter().all(|s| ordered.contains(s)), "les mêmes positions atteintes");
                checked += 1;
            }
            let mv = legal[rng.below(legal.len())].mv;
            k.play(&mut g, &mv).unwrap();
        }
    }
    assert!(checked >= 20, "trop peu de poses vérifiées : {checked}");
}
