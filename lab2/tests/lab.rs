//! Le banc d'essai lui-même : les coups proposés sont toujours jouables, une fin de tour se décide
//! jeton par jeton, ce que le joueur « sait » des pioches est exact. Et l'IA Expert telle que le jeu
//! l'emploie (`smallmobile_brain2::expert`).

use smallmobile_kernel::action::{Event, Journal, Logged};
use smallmobile_kernel::ids::{Power, Race};
use smallmobile_kernel::state::Stage;
use smallmobile_kernel::{Action, Game, Phase};
use smallmobile_lab2::agent::{Agent, Random};
use smallmobile_lab2::know::{Knowledge, Returns, determinize};
use smallmobile_lab2::moves::{self, Kind, Move};
use smallmobile_lab2::rand::Rng;
use smallmobile_lab2::table::Table;

/// Le journal d'une action pour ce que sait le joueur, gardé aussi en entier.
struct Both<'a>(&'a mut Returns, &'a mut Vec<Logged>);

impl Journal for Both<'_> {
    fn record(&mut self, entry: Logged) {
        self.0.record(entry);
        self.1.push(entry);
    }
}

/// Le réseau de départ du labo.
fn net_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("nets/gen0.bin")
}

/// L'Expert du labo, à `n` itérations (très peu : des parties d'essai rapides).
fn expert(n: usize) -> smallmobile_lab2::az::Az {
    smallmobile_lab2::agent::az_named(&format!("az:net={},n={n}", net_path().display())).unwrap()
}

/// Joue au hasard, coup par coup, jusqu'à la fin ; `visit` voit chaque position et ses coups.
fn random_game(
    g: &mut Game,
    k: &mut Knowledge,
    rng: &mut Rng,
    mut visit: impl FnMut(&Game, &Knowledge, &[moves::Choice]),
) {
    while g.state.phase != Phase::GameOver {
        let legal = moves::legal(g, k);
        visit(g, k, &legal);
        let m = legal[rng.below(legal.len())].mv;
        k.play(g, &m).unwrap_or_else(|e| panic!("{m:?} : {e:?}"));
    }
}

#[test]
fn every_legal_move_can_be_played() {
    let mut rng = Rng::new(7);
    let mut kinds = std::collections::HashSet::new();
    for game in 0..80u64 {
        let table = Table::new(2 + game as usize % 4, 500 + game);
        let (mut g, mut k) = (table.game(game as u32), Knowledge::default());
        random_game(&mut g, &mut k, &mut rng.fork(), |g, k, legal| {
            assert!(!legal.is_empty());
            let mut keys: Vec<_> = legal.iter().map(|c| c.key).collect();
            keys.sort();
            keys.dedup();
            assert_eq!(keys.len(), legal.len(), "deux coups de même clé");
            assert!(keys.iter().all(|&k| (k as usize) < moves::KEYS));
            for c in legal {
                let (mut w, mut kw) = (*g, *k);
                kw.play(&mut w, &c.mv).unwrap_or_else(|e| panic!("{:?} : {e:?}", c.mv));
                if let Move::Region(kind, _) = c.mv {
                    kinds.insert(kind);
                }
            }
        });
    }
    // toutes les sortes de coups sur une région se présentent dans ces parties
    assert_eq!(kinds.len(), moves::KINDS.len(), "{kinds:?}");
}

#[test]
fn a_redeploy_is_decided_one_token_at_a_time() {
    let mut rng = Rng::new(19);
    let (mut redeploys, mut powers) = (0, 0);
    for game in 0..40u64 {
        let table = Table::new(2 + game as usize % 4, 1900 + game);
        let (mut g, mut k) = (table.game(game as u32), Knowledge::default());
        while g.state.phase != Phase::GameOver {
            let legal = moves::legal(&g, &k);
            if legal.iter().any(|c| c.mv == Move::Lift) {
                assert_eq!(legal.len(), 1, "relever est forcé");
                let army = g.state.turn.army.unwrap();
                let own = g.owned(army);
                let total = g.state.race(army).hand + g.tokens_on_board(army);
                k.play(&mut g, &Move::Lift).unwrap();
                // un jeton par région, héros et campements relevés, rien de perdu
                assert!(own.iter().all(|r| g.state.regions[r].tokens == 1 && !g.state.regions[r].hero));
                assert_eq!(g.camps_on_board(army), 0);
                assert_eq!(g.state.race(army).hand, total - own.len() as i32);
                // puis pouvoirs, un jeton à la fois, et la fin du tour
                let mut tokens = 0;
                loop {
                    let legal = moves::legal(&g, &k);
                    let m = legal[rng.below(legal.len())].mv;
                    match m {
                        Move::Region(Kind::Token, _) => tokens += 1,
                        Move::Region(Kind::Hero | Kind::Camp | Kind::Fortress, _) => {
                            assert!(!k.redeploy.powers_done, "un pouvoir après un jeton");
                            powers += 1;
                        }
                        Move::End { .. } => {
                            assert_eq!(g.state.race(army).hand, 0, "tous les jetons posés");
                            assert_eq!(tokens, total - own.len() as i32);
                        }
                        other => panic!("coup inattendu en fin de tour : {other:?}"),
                    }
                    k.play(&mut g, &m).unwrap();
                    if g.state.phase != Phase::Redeploy || matches!(m, Move::End { .. }) {
                        break;
                    }
                }
                assert_eq!(k.redeploy, moves::Redeploy::default(), "suivi remis à zéro");
                redeploys += 1;
                continue;
            }
            let m = legal[rng.below(legal.len())].mv;
            k.play(&mut g, &m).unwrap();
        }
    }
    assert!(redeploys > 100 && powers > 0, "{redeploys} fins de tour, {powers} pouvoirs posés");
}

#[test]
fn powers_and_last_choices_follow_the_rules() {
    let mut rng = Rng::new(43);
    let (mut peaces, mut stouts) = (0, 0);
    for game in 0..120u64 {
        let table = Table::new(2 + game as usize % 4, 4300 + game);
        let (mut g, mut k) = (table.game(game as u32), Knowledge::default());
        random_game(&mut g, &mut k, &mut rng.fork(), |g, _, legal| {
            let s = &g.state;
            let (me, n) = (s.current, s.player_count());
            for c in legal {
                match c.mv {
                    // la cible de la paix, relativement au joueur qui finit son tour
                    Move::End { peace: Some(q), .. } => {
                        assert_eq!(c.key, moves::PEACE + ((q as usize + n - me) % n) as u16);
                        peaces += 1;
                    }
                    Move::End { stout: true, .. } => {
                        assert_eq!(c.key, moves::STOUT + 1);
                        stouts += 1;
                    }
                    Move::Region(Kind::Hero | Kind::Camp | Kind::Fortress, r) => {
                        let army = s.turn.army.unwrap();
                        assert_eq!(s.turn.stage, Stage::Active);
                        assert!(g.owned(army).contains(r as usize));
                        let power = s.race(army).power;
                        assert!([Power::Heroic, Power::Bivouacking, Power::Fortified].contains(&power));
                    }
                    _ => {}
                }
            }
        });
    }
    assert!(peaces > 0 && stouts > 0, "{peaces} paix, {stouts} déclins des Robustes proposés");
}

#[test]
fn a_whole_decision_compacts_to_the_same_state() {
    let mut rng = Rng::new(17);
    let mut units = 0;
    for game in 0..30u64 {
        let table = Table::new(2 + game as usize % 4, 1700 + game);
        let (mut g, mut k) = (table.game(game as u32), Knowledge::default());
        while g.state.phase != Phase::GameOver {
            let mut pick = rng.fork();
            let (actions, played) = moves::decision(&g, &k, |w, kw| {
                let legal = moves::legal(w, kw);
                legal[pick.below(legal.len())].mv
            })
            .unwrap();
            // les coups un à un, ou les actions groupées : le même état
            let (mut one, mut kone) = (g, k);
            for m in &played {
                kone.play(&mut one, m).unwrap();
            }
            let (mut grouped, mut kg) = (g, k);
            for a in &actions {
                kg.apply(&mut grouped, a).unwrap_or_else(|e| panic!("{a:?} : {e:?}"));
            }
            assert_eq!(grouped.state, one.state, "{played:?}");
            assert_eq!(kg.history, kone.history);
            if played.len() > 1 {
                units += 1;
                // au plus un placement, un ajout par région, et pas de jeton posé un à un
                let places = actions.iter().filter(|a| matches!(a, Action::Place(_))).count();
                let deploys: Vec<usize> = actions
                    .iter()
                    .filter_map(|a| if let Action::Deploy { region, .. } = a { Some(*region) } else { None })
                    .collect();
                let mut distinct = deploys.clone();
                distinct.dedup();
                assert!(places <= 1 && distinct.len() == deploys.len(), "{actions:?}");
                assert!(places == 0 || deploys.is_empty(), "{actions:?}");
            }
            g = grouped;
            k = kg;
        }
    }
    assert!(units > 100, "{units} fins de tour ou défenses");
}

#[test]
fn known_deck_bottom_is_exact() {
    let mut rng = Rng::new(3);
    let mut checked = 0;
    for game in 0..200u64 {
        let table = Table::new(2 + game as usize % 4, 900 + game);
        let mut g = table.game(game as u32);
        let mut k = Knowledge::default();
        // les peuples rendus sous la pioche, dans l'ordre, tels qu'un joueur les a vus passer
        let mut seen: Vec<Race> = Vec::new();
        while g.state.phase != Phase::GameOver {
            let actions = Random.act(&g, &k, &mut rng.fork());
            for a in &actions {
                let mut events: Vec<Logged> = Vec::new();
                let mut ret = k.journal(&g.state);
                g.apply(a, &mut Both(&mut ret, &mut events)).unwrap();
                for e in &events {
                    if let Event::Declined { removed, .. } = &e.event {
                        seen.extend(removed.iter().copied());
                    }
                }
                k.observe(ret, &g.state);
            }
            // le noyau suit lui-même ce fond connu (c'est ce que reçoit l'IA Expert dans le jeu)
            let public = Knowledge::of(&g.state);
            assert_eq!((public.race_tail, public.power_tail), (k.race_tail, k.power_tail));
            let deck = g.state.race_deck.as_slice();
            let tail = &deck[deck.len() - k.race_tail..];
            assert_eq!(tail, &seen[seen.len() - k.race_tail..]);
            checked += k.race_tail;
            // un monde possible garde le marché, le fond connu et le contenu de la pioche
            let w = determinize(&g, &k, &mut rng);
            assert_eq!(w.state.market, g.state.market);
            assert_eq!(&w.state.race_deck.as_slice()[deck.len() - k.race_tail..], tail);
            let mut a = w.state.race_deck.as_slice().to_vec();
            let mut b = deck.to_vec();
            a.sort();
            b.sort();
            assert_eq!(a, b);
        }
    }
    assert!(checked > 0, "aucune carte rendue dans ces parties : le test ne vérifie rien");
}

#[test]
fn the_network_sees_only_what_the_player_knows() {
    use smallmobile_lab2::encode::encode;
    let mut rng = Rng::new(11);
    for game in 0..40u64 {
        let table = Table::new(2 + game as usize % 4, 1300 + game);
        let (mut g, mut k) = (table.game(game as u32), Knowledge::default());
        let mut worlds = rng.fork();
        random_game(&mut g, &mut k, &mut rng.fork(), |g, k, _| {
            // deux mondes possibles donnent exactement la même entrée au réseau
            let a = encode(&determinize(g, k, &mut worlds), k);
            let b = encode(&determinize(g, k, &mut worlds), k);
            assert!(a.node == b.node && a.global == b.global, "l'encodage dépend de ce qui est caché");
            // « moi » est le joueur qui agit : ses pièces sont en tête des joueurs
            let me = g.state.actor().unwrap();
            assert_eq!(a.global[67 + 1] as i32, g.state.players[me].coins.clamp(0, 255));
        });
    }
}

#[test]
fn the_fast_network_computes_like_the_simple_one() {
    use smallmobile_lab2::encode::encode;
    let net = smallmobile_lab2::net::Net::load(&net_path()).unwrap();
    let close = |a: f32, b: f32| a == b || (a - b).abs() <= 1e-3 * (1.0 + b.abs());
    let mut rng = Rng::new(31);
    let mut checked = 0;
    for game in 0..8u64 {
        let table = Table::new(2 + game as usize % 4, 3100 + game);
        let (mut g, mut k) = (table.game(game as u32), Knowledge::default());
        random_game(&mut g, &mut k, &mut rng.fork(), |g, k, _| {
            let e = encode(g, k);
            let fast = net.eval(&e, g.map);
            let simple = net.eval_portable(&e, |i| &g.map.regions[i].adj);
            let pairs = fast.win.iter().chain(&fast.gain).chain(&fast.logits);
            for (&a, &b) in pairs.zip(simple.win.iter().chain(&simple.gain).chain(&simple.logits)) {
                assert!(close(a, b), "calcul rapide {a}, calcul simple {b}");
            }
            checked += 1;
        });
    }
    assert!(checked > 500, "{checked} positions");
}

#[test]
fn the_search_sees_only_what_the_player_knows() {
    use smallmobile_kernel::Level;
    use smallmobile_lab2::agent::KernelAi;
    let az = expert(30);
    let hard = KernelAi::new(Level::Hard);
    let mut rng = Rng::new(23);
    let mut checked = 0;
    for game in 0..4u64 {
        let table = Table::new(2 + game as usize % 4, 2300 + game);
        let (mut g, mut k) = (table.game(game as u32), Knowledge::default());
        let mut other_rng = rng.fork();
        random_game(&mut g, &mut k, &mut rng.fork(), |g, k, legal| {
            if legal.len() > 1 && other_rng.below(12) == 0 {
                // la même partie vue par le joueur, mais d'autres pioches et d'autres dés cachés :
                // à graine égale, chaque joueur doit y faire exactement la même chose
                let other = determinize(g, k, &mut other_rng);
                for explore in [false, true] {
                    let a = az.search(g, k, &mut Rng::new(checked), explore);
                    let b = az.search(&other, k, &mut Rng::new(checked), explore);
                    assert_eq!(a.visits, b.visits, "l'Expert voit ce qui est caché");
                    assert_eq!(a.value.to_bits(), b.value.to_bits());
                }
                let a = hard.act(g, k, &mut Rng::new(checked));
                assert_eq!(a, hard.act(&other, k, &mut Rng::new(checked)), "l'IA Difficile voit ce qui est caché");
                checked += 1;
            }
        });
    }
    assert!(checked > 10, "trop peu de positions vérifiées : {checked}");
}

#[test]
fn the_expert_plays_whole_games() {
    use smallmobile_brain2::expert::Expert;
    use smallmobile_kernel::Level;
    let net = std::sync::Arc::new(smallmobile_lab2::net::Net::load(&net_path()).unwrap());
    let expert = Expert::new(net, 8);
    for players in [2, 5] {
        let table = Table::new(players, 2100 + players as u64);
        let mut g = table.game(players as u32);
        g.state.players[0].level = Level::Expert;
        let (mut decisions, mut redeploys) = (0, 0);
        while g.state.phase != Phase::GameOver {
            let actions = if g.state.actor() == Some(0) {
                decisions += 1;
                let d = expert.decide(&g, decisions).unwrap();
                assert!(!d.actions.is_empty());
                redeploys += usize::from(d.actions.iter().any(|a| matches!(a, Action::EndTurn { .. })));
                d.actions
            } else {
                // l'IA du noyau joue l'Expert comme Difficile ; ici les autres places
                let tuning = smallmobile_kernel::Tuning::default();
                let ai = smallmobile_kernel::Ai::new(&tuning);
                vec![ai.decide(&g, smallmobile_kernel::Restrictions::default()).unwrap()]
            };
            for a in actions {
                g.apply(&a, &mut smallmobile_kernel::NoJournal).unwrap_or_else(|e| panic!("{a:?} : {e:?}"));
            }
        }
        assert!(decisions > 10 && redeploys > 3, "{decisions} décisions, {redeploys} fins de tour");
    }
}

#[test]
fn the_lab1_expert_plays_in_the_arena() {
    use smallmobile_lab2::agent::by_name;
    use smallmobile_lab2::arena::{self, Config, Lineup};
    let lab1 = by_name(&format!("lab1:net={}/../lab1/nets/gen11.bin,n=4", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let az = expert(4);
    let lineups = vec![Lineup::hero(2, &az, lab1.as_ref())];
    let run = arena::run(&Config { deals: 1, threads: 2, seed: 2, stop: None }, &lineups);
    assert_eq!(run.results.len(), 4, "deux parties, deux places");
}

#[test]
fn game_ai_styles_play_differently() {
    use smallmobile_kernel::Level;
    use smallmobile_lab2::agent::{KernelAi, STYLES};
    use smallmobile_lab2::know::habit;
    assert!(KernelAi::style(Level::Normal, "foe=1,nothing=2").is_none(), "réglage inconnu");
    // par style : régions prises aux autres joueurs, déclins, manches tenues par peuple décliné (par
    // partie), part des prises faites au plus riche
    let profile = |ai: &KernelAi| {
        let mut rng = Rng::new(17);
        let (mut taken, mut declines, mut age, mut leader) = (0.0, 0.0, 0.0, 0.0);
        let games = 30;
        for game in 0..games {
            let table = Table::new(3, 1700 + game);
            let mut g = table.game(game as u32);
            let mut k = Knowledge::default();
            while g.state.phase != Phase::GameOver {
                for a in ai.act(&g, &k, &mut rng.fork()) {
                    k.apply(&mut g, &a).unwrap();
                }
            }
            for x in &k.history.habits[..3] {
                taken += (x[habit::ACTIVE] + x[habit::DECLINED]) as f64 / games as f64;
                declines += x[habit::DECLINES] as f64 / games as f64;
                age += x[habit::DECLINE_AGE] as f64 / games as f64;
                leader += x[habit::LEADER] as f64 / games as f64;
            }
        }
        (taken, declines, age / declines, leader / taken)
    };
    let normal = profile(&KernelAi::new(Level::Normal));
    let styles: Vec<_> =
        STYLES.iter().map(|(name, _)| profile(&KernelAi::style(Level::Normal, name).unwrap())).collect();
    for ((name, _), p) in STYLES.iter().zip(&styles) {
        eprintln!(
            "{name:>10} : {:.1} régions prises aux autres ({:.0} % au plus riche), {:.1} déclins, {:.1} manches par peuple décliné",
            p.0,
            100.0 * p.3,
            p.1,
            p.2
        );
    }
    eprintln!("{:>10} : {:.1} ({:.0} %), {:.1}, {:.1}", "normal", normal.0, 100.0 * normal.3, normal.1, normal.2);
    let [aggressive, hunter, prudent, early, late] = styles[..] else { unreachable!() };
    assert!(aggressive.0 > normal.0 * 1.05, "l'agressive prend plus aux autres");
    assert!(hunter.3 > normal.3 + 0.03, "la chasseuse vise davantage le plus riche");
    assert!(prudent.0 < normal.0 * 0.8, "la prudente prend moins aux autres");
    assert!(
        early.1 > late.1 * 1.2 && early.2 < late.2,
        "décliner tôt : plus de déclins, des peuples tenus moins longtemps"
    );
}

#[test]
fn the_arena_puts_every_player_at_every_seat() {
    use smallmobile_kernel::Level;
    use smallmobile_lab2::agent::KernelAi;
    use smallmobile_lab2::arena::{self, Config, Lineup};
    let (easy, normal) = (KernelAi::new(Level::Easy), KernelAi::new(Level::Normal));
    let cfg = Config { deals: 6, threads: 3, seed: 4, stop: None };
    let lineups: Vec<Lineup> = [2, 3, 4].into_iter().map(|n| Lineup::hero(n, &easy, &normal)).collect();
    let run = arena::run(&cfg, &lineups);
    // donnes à 2, 3, 4, 2, 3, 4 joueurs ; une partie par rotation, une ligne par place
    assert_eq!(run.results.len(), 2 * (4 + 9 + 16));
    for r in &run.results {
        assert_eq!(r.role == 0, r.seat == r.shift, "{r:?}");
    }
    for d in 0..cfg.deals {
        for shift in 0..[2, 3, 4][d % 3] {
            let game = run.results.iter().filter(|r| r.deal == d && r.shift == shift);
            assert!((game.map(|r| r.win).sum::<f64>() - 1.0).abs() < 1e-9, "une victoire par partie");
        }
    }
    // même graine, mêmes parties
    let key = |run: &arena::Run| -> Vec<_> {
        run.results.iter().map(|r| (r.deal, r.shift, r.seat, r.role, r.coins, r.rank)).collect()
    };
    assert_eq!(key(&arena::run(&cfg, &lineups)), key(&run));
}

#[test]
fn the_arena_score_weighs_sizes_equally() {
    use smallmobile_lab2::arena::{score, verdict};
    // 2 joueurs : 10 et 30 (moyenne 20, variance 200) ; 3 joueurs : 0, 0 et 6 (moyenne 2, variance 12)
    let (s, se) = score(&[(2, 10.0), (3, 0.0), (2, 30.0), (3, 0.0), (3, 6.0)]).unwrap();
    assert!((s - 11.0).abs() < 1e-9 && (se - (200.0 / 2.0 + 12.0 / 3.0f64).sqrt() / 2.0).abs() < 1e-9);
    assert_eq!(score(&[(2, 10.0), (2, 30.0), (3, 0.0)]), None, "une seule donne à 3 joueurs");
    assert_eq!(verdict(s, se, 3.0), None);
    assert_eq!(verdict(6.0, 0.8, 3.0), Some(true));
    assert_eq!(verdict(0.0, 0.8, 3.0), Some(false));
}

#[test]
fn a_clear_arena_stops_early_and_keeps_its_first_deals() {
    use smallmobile_kernel::Level;
    use smallmobile_lab2::agent::KernelAi;
    use smallmobile_lab2::arena::{self, Config, Lineup, Sequential};
    let (normal, easy) = (KernelAi::new(Level::Normal), KernelAi::new(Level::Easy));
    let lineups: Vec<Lineup> = [2, 3].into_iter().map(|n| Lineup::hero(n, &normal, &easy)).collect();
    let stop = Some(Sequential { role: 0, delta: 10.0, look: 2 });
    let stopped = arena::run(&Config { deals: 60, threads: 4, seed: 9, stop }, &lineups);
    assert_eq!(stopped.verdict, Some(true), "Normal bat nettement Facile");
    assert!(stopped.deals < 60 && stopped.deals.is_multiple_of(4), "{} donnes", stopped.deals);
    let full = arena::run(&Config { deals: 60, threads: 4, seed: 9, stop: None }, &lineups);
    assert_eq!((full.deals, full.verdict), (60, None));
    let key = |r: &arena::SeatResult| (r.deal, r.shift, r.seat, r.role, r.coins, r.rank);
    let first: Vec<_> = full.results.iter().filter(|r| r.deal < stopped.deals).map(key).collect();
    assert_eq!(stopped.results.iter().map(key).collect::<Vec<_>>(), first, "les mêmes premières donnes, entières");
}

#[test]
fn varied_tables_record_only_the_learner() {
    use smallmobile_kernel::Level;
    use smallmobile_lab2::agent::{Agent, KernelAi};
    use smallmobile_lab2::selfplay::{self, Field, GameLog, ILLEGAL, PROBABILITY, RECORD};
    let (easy, hard) = (KernelAi::new(Level::Easy), KernelAi::new(Level::Hard));
    let field = Field { agents: vec![&easy as &dyn Agent, &hard], share: 0.5 };
    let az = expert(2);
    let dir = std::env::temp_dir().join(format!("smallmobile-field-{}", std::process::id()));
    let opts = selfplay::Options { full_share: 1.0, fast: 1 };
    let cfg = selfplay::Config { players: vec![2, 3, 4, 5], games: 12, threads: 2, seed: 12, opts };
    selfplay::run(&cfg, &az, &field, &dir).unwrap();
    let (mut mixed, mut plain) = (0, 0);
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if !path.to_string_lossy().ends_with(selfplay::SUFFIX) {
            continue;
        }
        let logs: Vec<GameLog> = std::fs::read_to_string(selfplay::moves_file(&path))
            .unwrap()
            .lines()
            .map(|l| GameLog::parse(l).unwrap())
            .collect();
        let bytes = std::fs::read(&path).unwrap();
        let mut records = bytes.chunks(RECORD);
        for log in &logs {
            // une position par décision de l'IA qui apprend, marquée « table variée » s'il y a
            // lieu, avec la politique améliorée sur ses coups possibles
            let table = log.table();
            log.replay(&table, |g, k| {
                let me = g.state.actor().unwrap();
                let legal = moves::legal(g, k);
                if legal.len() > 1 && log.seats.get(me).is_none_or(|&s| s < 0) {
                    let rec = records.next().unwrap();
                    assert_eq!(rec[2], !log.seats.is_empty() as u8);
                    let policy: Vec<u16> =
                        rec[43..43 + 2 * moves::KEYS].chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
                    let legal_keys: Vec<usize> = legal.iter().map(|c| c.key as usize).collect();
                    let mut sum = 0.0;
                    for (key, &p) in policy.iter().enumerate() {
                        assert_eq!(p == ILLEGAL, !legal_keys.contains(&key), "coup {key}");
                        if p != ILLEGAL {
                            sum += p as f32;
                        }
                    }
                    assert!((sum - PROBABILITY).abs() < legal.len() as f32, "probabilités : {sum}");
                }
            })
            .unwrap();
            if log.seats.is_empty() {
                plain += 1;
            } else {
                assert!(log.seats.contains(&-1) && log.seats.iter().any(|&s| s >= 0));
                mixed += 1;
            }
            assert_eq!(GameLog::parse(&log.line()).unwrap(), *log);
        }
        assert!(records.next().is_none());
    }
    assert!(mixed > 0 && plain > 0, "{mixed} tables variées, {plain} autres");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn position_files_say_how_many_positions_each_game_has() {
    use smallmobile_lab2::selfplay::{self, Field};
    let dir = std::env::temp_dir().join(format!("smallmobile-games-{}", std::process::id()));
    let opts = selfplay::Options { full_share: 1.0, fast: 1 };
    let cfg = selfplay::Config { players: vec![2, 3], games: 5, threads: 2, seed: 3, opts };
    selfplay::run(&cfg, &expert(2), &Field::none(), &dir).unwrap();
    let (mut total, mut positions) = (0, 0);
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.to_string_lossy().ends_with(selfplay::SUFFIX) {
            let counts: Vec<u32> = std::fs::read(selfplay::games_file(&path))
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            let size = std::fs::metadata(&path).unwrap().len() as usize;
            assert_eq!(counts.iter().sum::<u32>() as usize * selfplay::RECORD, size, "{}", path.display());
            total += counts.len();
            positions += size / selfplay::RECORD;
        }
    }
    assert_eq!(total, 5, "une ligne par partie");
    assert!(positions > 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn self_play_games_can_be_replayed() {
    use smallmobile_lab2::selfplay::{self, Field, GameLog};
    use smallmobile_lab2::table::Outcome;
    let dir = std::env::temp_dir().join(format!("smallmobile-replay-{}", std::process::id()));
    let opts = selfplay::Options { full_share: 1.0, fast: 1 };
    let cfg = selfplay::Config { players: vec![2, 4], games: 4, threads: 2, seed: 5, opts };
    selfplay::run(&cfg, &expert(2), &Field::none(), &dir).unwrap();
    let mut games = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if !path.to_string_lossy().ends_with(selfplay::SUFFIX) {
            continue;
        }
        let counts: Vec<u32> = std::fs::read(selfplay::games_file(&path))
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let text = std::fs::read_to_string(selfplay::moves_file(&path)).unwrap();
        let logs: Vec<GameLog> = text.lines().map(|l| GameLog::parse(l).expect("ligne lisible")).collect();
        assert_eq!(logs.len(), counts.len());
        for (log, &count) in logs.iter().zip(&counts) {
            assert_eq!(GameLog::parse(&log.line()).as_ref(), Some(log));
            let table = log.table();
            // une position enregistrée par coup où il y avait le choix, et le même résultat
            let mut choices = 0;
            let end = log.replay(&table, |g, k| choices += usize::from(moves::legal(g, k).len() > 1)).unwrap();
            assert_eq!(choices, count as usize);
            assert_eq!(&Outcome::of(&end).coins[..log.players], &log.coins[..]);
            games += 1;
        }
    }
    assert_eq!(games, 4);
    std::fs::remove_dir_all(&dir).unwrap();
}
