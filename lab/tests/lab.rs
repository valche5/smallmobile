//! Le banc d'essai lui-même : les coups proposés sont toujours jouables, et ce que le joueur « sait »
//! des pioches est exact. Et l'IA Expert telle que le jeu l'emploie (`smallmobile_brain::expert`).

use smallmobile_kernel::Phase;
use smallmobile_kernel::action::{Event, Logged};
use smallmobile_kernel::ids::Race;
use smallmobile_lab::know::{Knowledge, Returns, determinize};
use smallmobile_lab::moves;
use smallmobile_lab::rand::Rng;
use smallmobile_lab::table::Table;

#[test]
fn every_legal_move_can_be_played() {
    let mut rng = Rng::new(7);
    for game in 0..60u64 {
        let table = Table::new(2 + game as usize % 4, 500 + game);
        let mut g = table.game(game as u32);
        while g.state.phase != Phase::GameOver {
            let legal = moves::legal(&g);
            assert!(!legal.is_empty());
            let mut keys: Vec<_> = legal.iter().map(|c| c.key).collect();
            keys.sort();
            keys.dedup();
            assert_eq!(keys.len(), legal.len(), "deux coups de même clé");
            assert!(keys.iter().all(|&k| (k as usize) < moves::KEYS));
            for c in &legal {
                let mut w = g;
                let m = c.mv;
                moves::play(&mut w, &m, &mut Returns::default()).unwrap_or_else(|e| panic!("{m:?} : {e:?}"));
            }
            let m = legal[rng.below(legal.len())].mv;
            moves::play(&mut g, &m, &mut Returns::default()).unwrap();
        }
    }
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
            let legal = moves::legal(&g);
            let m = legal[rng.below(legal.len())].mv;
            let mut events: Vec<Logged> = Vec::new();
            moves::play(&mut g, &m, &mut events).unwrap();
            let returns: usize = events
                .iter()
                .map(|e| match &e.event {
                    Event::Declined { removed, .. } => {
                        seen.extend(removed.iter().copied());
                        removed.len()
                    }
                    _ => 0,
                })
                .sum();
            k.observe(returns, &g.state);
            // le noyau suit lui-même ce fond connu (c'est ce que reçoit l'IA Expert dans le jeu)
            assert_eq!(Knowledge::of(&g.state), k);
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
    use smallmobile_lab::encode::encode;
    let mut rng = Rng::new(11);
    for game in 0..40u64 {
        let table = Table::new(2 + game as usize % 4, 1300 + game);
        let mut g = table.game(game as u32);
        let mut k = Knowledge::default();
        while g.state.phase != Phase::GameOver {
            // deux mondes possibles donnent exactement la même entrée au réseau
            let a = encode(&determinize(&g, &k, &mut rng), k.race_tail);
            let b = encode(&determinize(&g, &k, &mut rng), k.race_tail);
            assert!(a.node == b.node && a.global == b.global, "l'encodage dépend de ce qui est caché");
            // « moi » est le joueur qui agit : ses pièces sont en tête des joueurs
            let me = g.state.actor().unwrap();
            assert_eq!(a.global[67 + 1] as i32, g.state.players[me].coins.clamp(0, 255));
            let legal = moves::legal(&g);
            let m = legal[rng.below(legal.len())].mv;
            let mut ret = Returns::default();
            moves::play(&mut g, &m, &mut ret).unwrap();
            k.observe(ret.0, &g.state);
        }
    }
}

#[test]
fn old_positions_can_be_reencoded() {
    use smallmobile_kernel::Level;
    use smallmobile_lab::agent::KernelAi;
    use smallmobile_lab::encode::{GLOBAL, GLOBAL_V1, NODE, REGIONS};
    use smallmobile_lab::moves::{FINISH, KEYS};
    use smallmobile_lab::selfplay::{self, GameLog, ILLEGAL, RECORD};
    let dir = std::env::temp_dir().join(format!("smallmobile-reencode-{}", std::process::id()));
    // des positions d'aujourd'hui (Diplomates compris)…
    let opts = selfplay::Options { sample_rounds: 2, full_share: 1.0, fast: 1 };
    let cfg = selfplay::Config { players: vec![2, 3, 4, 5], games: 12, threads: 2, seed: 9, opts };
    selfplay::run(&cfg, &KernelAi::new(Level::Normal), &dir.join("new")).unwrap();
    // … remises au format de la version 3 : entrées de la version 1, clés de paix par place absolue
    let (keys_at, node_at) = (44, 44 + KEYS * 2 + REGIONS * 8);
    let global_at = node_at + REGIONS * NODE;
    let mut logs = Vec::new();
    let mut originals = Vec::new();
    let mut peaces = 0;
    std::fs::create_dir_all(dir.join("old")).unwrap();
    for entry in std::fs::read_dir(dir.join("new")).unwrap() {
        let path = entry.unwrap().path();
        if !path.to_string_lossy().ends_with(selfplay::SUFFIX) {
            continue;
        }
        let games: Vec<GameLog> = std::fs::read_to_string(selfplay::moves_file(&path))
            .unwrap()
            .lines()
            .map(|l| GameLog::parse(l).unwrap())
            .collect();
        let bytes = std::fs::read(&path).unwrap();
        let mut records = bytes.chunks(RECORD);
        let mut old = Vec::new();
        for log in &games {
            let table = log.table();
            log.replay(&table, |g, _| {
                if moves::legal(g).len() < 2 {
                    return;
                }
                let rec = records.next().unwrap();
                let (seat, n) = (g.state.actor().unwrap(), g.state.player_count());
                let mut v3 = rec[..node_at].to_vec();
                for plan in 0..6 {
                    let base = keys_at + 2 * (FINISH as usize + plan * 12);
                    let relative = rec[base..base + 24].to_vec();
                    peaces += relative[4..].chunks(2).filter(|v| v != &ILLEGAL.to_le_bytes()).count();
                    for slot in 1..6 {
                        for stout in 0..2 {
                            let target = (slot + n - 1 - seat) % n;
                            let from = if slot <= n && target != 0 { 2 * (target * 2 + stout) } else { usize::MAX };
                            let at = base + 2 * (slot * 2 + stout);
                            let v = if from == usize::MAX {
                                ILLEGAL.to_le_bytes()
                            } else {
                                [relative[from], relative[from + 1]]
                            };
                            v3[at..at + 2].copy_from_slice(&v);
                        }
                    }
                }
                v3.extend_from_slice(&rec[node_at..global_at + GLOBAL_V1]);
                v3.extend_from_slice(&rec[global_at + GLOBAL..]);
                old.extend(v3);
            })
            .unwrap();
        }
        assert!(records.next().is_none(), "une position par décision");
        let name = path.file_name().unwrap().to_string_lossy().replace(selfplay::SUFFIX, ".v3.bin");
        let target = dir.join("old").join(name);
        std::fs::write(&target, old).unwrap();
        std::fs::copy(selfplay::games_file(&path), selfplay::games_file(&target)).unwrap();
        std::fs::copy(selfplay::moves_file(&path), selfplay::moves_file(&target)).unwrap();
        logs.extend(games);
        originals.push((path.file_name().unwrap().to_owned(), bytes));
    }
    // le réencodage redonne les positions d'aujourd'hui, octet pour octet
    let (files, positions) = smallmobile_lab::reencode::run(&logs, &dir.join("old"), &dir.join("again"), 2).unwrap();
    assert_eq!(files, originals.len());
    assert!(positions > 100 && peaces > 0, "{positions} positions, {peaces} clés de paix possibles");
    for (name, bytes) in originals {
        assert!(std::fs::read(dir.join("again").join(&name)).unwrap() == bytes, "{name:?}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn peace_keys_name_the_target_relative_to_the_player() {
    let mut rng = Rng::new(41);
    let mut peaces = 0;
    for game in 0..80u64 {
        let table = Table::new(2 + game as usize % 4, 4100 + game);
        let mut g = table.game(game as u32);
        let mut k = Knowledge::default();
        while g.state.phase != Phase::GameOver {
            let legal = moves::legal(&g);
            let keys: std::collections::HashSet<u16> = legal.iter().map(|c| c.key).collect();
            assert_eq!(keys.len(), legal.len(), "une clé par coup");
            let (me, n) = (g.state.current, g.state.player_count());
            for c in &legal {
                if let moves::Move::Finish(f) = c.mv {
                    let target = f.peace.map_or(0, |q| (q + n - me) % n);
                    assert_eq!((c.key - moves::FINISH) % 12, target as u16 * 2 + u16::from(f.stout));
                    peaces += f.peace.is_some() as usize;
                }
            }
            let m = legal[rng.below(legal.len())].mv;
            let mut ret = Returns::default();
            moves::play(&mut g, &m, &mut ret).unwrap();
            k.observe(ret.0, &g.state);
        }
    }
    assert!(peaces > 0);
}

#[test]
fn the_fast_network_computes_like_the_simple_one() {
    use smallmobile_lab::encode::encode;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("nets/gen8.bin");
    let net = smallmobile_lab::net::Net::load(&path).unwrap();
    let close = |a: f32, b: f32| a == b || (a - b).abs() <= 1e-3 * (1.0 + b.abs());
    let mut rng = Rng::new(31);
    let mut checked = 0;
    for game in 0..8u64 {
        let table = Table::new(2 + game as usize % 4, 3100 + game);
        let mut g = table.game(game as u32);
        let mut k = Knowledge::default();
        while g.state.phase != Phase::GameOver {
            let e = encode(&g, k.race_tail);
            let fast = net.eval(&e, g.map);
            let simple = net.eval_portable(&e, |i| &g.map.regions[i].adj);
            let pairs = fast.win.iter().chain(&fast.gain).chain(&fast.logits);
            for (&a, &b) in pairs.zip(simple.win.iter().chain(&simple.gain).chain(&simple.logits)) {
                assert!(close(a, b), "calcul rapide {a}, calcul simple {b}");
            }
            checked += 1;
            let legal = moves::legal(&g);
            let m = legal[rng.below(legal.len())].mv;
            let mut ret = Returns::default();
            moves::play(&mut g, &m, &mut ret).unwrap();
            k.observe(ret.0, &g.state);
        }
    }
    assert!(checked > 500, "{checked} positions");
}

#[test]
fn the_search_sees_only_what_the_player_knows() {
    use smallmobile_kernel::Level;
    use smallmobile_lab::agent::{Agent, KernelAi};
    use smallmobile_lab::az::{Az, Config};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("nets/gen8.bin");
    let net = std::sync::Arc::new(smallmobile_lab::net::Net::load(&path).unwrap());
    let shipped = Az::new(smallmobile_brain::expert::config(30), net.clone(), "gen8".into());
    let alone = Az::new(Config { rollout: 0.0, ..shipped.cfg }, net, "gen8".into());
    let hard = KernelAi::new(Level::Hard);
    let mut rng = Rng::new(23);
    let mut checked = 0;
    for game in 0..4u64 {
        let table = Table::new(2 + game as usize % 4, 2300 + game);
        let mut g = table.game(game as u32);
        let mut k = Knowledge::default();
        while g.state.phase != Phase::GameOver {
            let legal = moves::legal(&g);
            if legal.len() > 1 && rng.below(12) == 0 {
                // la même partie vue par le joueur, mais d'autres pioches et d'autres dés cachés :
                // à graine égale, chaque joueur doit y faire exactement la même chose
                let other = determinize(&g, &k, &mut rng);
                for az in [&shipped, &alone] {
                    let a = az.search(&g, &k, &mut Rng::new(checked));
                    let b = az.search(&other, &k, &mut Rng::new(checked));
                    assert_eq!(a.visits, b.visits, "{} voit ce qui est caché", az.describe());
                    assert_eq!(a.value.to_bits(), b.value.to_bits(), "{}", az.describe());
                }
                let a = hard.act(&g, &k, &mut Rng::new(checked));
                assert_eq!(a, hard.act(&other, &k, &mut Rng::new(checked)), "l'IA Difficile voit ce qui est caché");
                checked += 1;
            }
            let m = legal[rng.below(legal.len())].mv;
            let mut ret = Returns::default();
            moves::play(&mut g, &m, &mut ret).unwrap();
            k.observe(ret.0, &g.state);
        }
    }
    assert!(checked > 10, "trop peu de positions vérifiées : {checked}");
}

#[test]
fn the_arena_puts_every_player_at_every_seat() {
    use smallmobile_kernel::Level;
    use smallmobile_lab::agent::KernelAi;
    use smallmobile_lab::arena::{self, Config, Lineup};
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
    use smallmobile_lab::arena::{score, verdict};
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
    use smallmobile_lab::agent::KernelAi;
    use smallmobile_lab::arena::{self, Config, Lineup, Sequential};
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
fn position_files_say_how_many_positions_each_game_has() {
    use smallmobile_kernel::Level;
    use smallmobile_lab::agent::KernelAi;
    use smallmobile_lab::{label, selfplay};
    let dir = std::env::temp_dir().join(format!("smallmobile-games-{}", std::process::id()));
    let opts = selfplay::Options { sample_rounds: 2, full_share: 1.0, fast: 1 };
    let cfg = selfplay::Config { players: vec![2, 3], games: 5, threads: 2, seed: 3, opts };
    selfplay::run(&cfg, &KernelAi::new(Level::Normal), &dir.join("play")).unwrap();
    let cfg = label::Config {
        players: vec![2, 5],
        games: 4,
        threads: 2,
        seed: 3,
        share: 0.1,
        rollouts: 2,
        random: 0.1,
        judge: label::Judge::Normal,
    };
    label::run(&cfg, None, &dir.join("label")).unwrap();
    for (sub, games) in [("play", 5), ("label", 4)] {
        let (mut total, mut positions) = (0, 0);
        for entry in std::fs::read_dir(dir.join(sub)).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "bin") {
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
        assert_eq!(total, games, "{sub} : une ligne par partie");
        assert!(positions > 0);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn self_play_games_can_be_replayed() {
    use smallmobile_kernel::Level;
    use smallmobile_lab::agent::KernelAi;
    use smallmobile_lab::selfplay::{self, GameLog};
    use smallmobile_lab::table::Outcome;
    let dir = std::env::temp_dir().join(format!("smallmobile-replay-{}", std::process::id()));
    let opts = selfplay::Options { sample_rounds: 2, full_share: 1.0, fast: 1 };
    let cfg = selfplay::Config { players: vec![2, 4], games: 4, threads: 2, seed: 5, opts };
    selfplay::run(&cfg, &KernelAi::new(Level::Normal), &dir).unwrap();
    let mut games = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "bin") {
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
            let end = log.replay(&table, |g, _| choices += usize::from(moves::legal(g).len() > 1)).unwrap();
            assert_eq!(choices, count as usize);
            assert_eq!(&Outcome::of(&end).coins[..log.players], &log.coins[..]);
            games += 1;
        }
    }
    assert_eq!(games, 4);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn played_games_can_be_judged_again_by_the_network() {
    use smallmobile_kernel::Level;
    use smallmobile_lab::agent::{KernelAi, az_named};
    use smallmobile_lab::label::{self, Judge};
    use smallmobile_lab::selfplay::{self, GameLog};
    let dir = std::env::temp_dir().join(format!("smallmobile-judge-{}", std::process::id()));
    let opts = selfplay::Options { sample_rounds: 2, full_share: 1.0, fast: 1 };
    let cfg = selfplay::Config { players: vec![3, 5], games: 2, threads: 1, seed: 8, opts };
    selfplay::run(&cfg, &KernelAi::new(Level::Normal), &dir.join("play")).unwrap();
    let mut logs = Vec::new();
    for entry in std::fs::read_dir(dir.join("play")).unwrap() {
        let path = entry.unwrap().path();
        if path.to_string_lossy().ends_with(".moves") {
            logs.extend(std::fs::read_to_string(&path).unwrap().lines().map(|l| GameLog::parse(l).unwrap()));
        }
    }
    let net = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("nets/gen8.bin");
    let judge = Judge::Net(Box::new(az_named(&format!("az:net={}", net.display())).unwrap()));
    label::rejudge(&logs, 0.1, 2, &judge, 1, 2, &dir.join("judged")).unwrap();
    let mut positions = 0;
    for entry in std::fs::read_dir(dir.join("judged")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "bin") {
            for rec in std::fs::read(&path).unwrap().as_chunks::<{ selfplay::RECORD }>().0 {
                // la part de victoire de chaque joueur, d'après des fins jouées jusqu'au bout
                let win: f32 = (0..rec[0] as usize)
                    .map(|r| f32::from_le_bytes(rec[4 + 4 * r..8 + 4 * r].try_into().unwrap()))
                    .sum();
                assert!((win - 1.0).abs() < 1e-4, "parts de victoire : {win}");
                positions += 1;
            }
        }
    }
    assert!(positions > 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn known_seats_give_certain_beliefs() {
    use smallmobile_brain::opponents::{Beliefs, Model};
    let b = Beliefs::known(&[Model::Net, Model::Normal, Model::Net]);
    let mut rng = Rng::new(1);
    for (q, model) in [Model::Net, Model::Normal, Model::Net].into_iter().enumerate() {
        assert!((0..50).all(|_| b.sample(q, &mut rng) == model));
    }
}

#[test]
fn a_move_cut_into_actions_plays_the_same() {
    let mut rng = Rng::new(17);
    for game in 0..30u64 {
        let table = Table::new(2 + game as usize % 4, 1700 + game);
        let mut g = table.game(game as u32);
        while g.state.phase != Phase::GameOver {
            let legal = moves::legal(&g);
            for c in &legal {
                let mut played = g;
                moves::play(&mut played, &c.mv, &mut Returns::default()).unwrap();
                let mut cut = g;
                for a in moves::actions(&g, &c.mv).unwrap() {
                    cut.apply(&a, &mut smallmobile_kernel::NoJournal).unwrap_or_else(|e| panic!("{a:?} : {e:?}"));
                }
                assert_eq!(cut.state, played.state, "{:?}", c.mv);
            }
            let m = legal[rng.below(legal.len())].mv;
            moves::play(&mut g, &m, &mut Returns::default()).unwrap();
        }
    }
}

#[test]
fn the_expert_plays_whole_games() {
    use smallmobile_brain::expert::Expert;
    use smallmobile_kernel::Level;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("nets/gen8.bin");
    let net = std::sync::Arc::new(smallmobile_lab::net::Net::load(&path).unwrap());
    let expert = Expert::new(net, 8);
    for players in [2, 5] {
        let table = Table::new(players, 2100 + players as u64);
        let mut g = table.game(players as u32);
        g.state.players[0].level = Level::Expert;
        let mut decisions = 0;
        while g.state.phase != Phase::GameOver {
            let actions = if g.state.actor() == Some(0) {
                decisions += 1;
                let d = expert.decide(&g, decisions).unwrap();
                assert!(!d.actions.is_empty());
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
        assert!(decisions > 10);
    }
}
