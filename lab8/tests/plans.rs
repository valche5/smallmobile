//! Les plans : le décodeur joue des parties entières sans coup impossible ; les parties d'imitation
//! se découpent en plans dont les cibles se tiennent.

use std::path::Path;

use smallmobile_lab7::agent::{self, Agent};
use smallmobile_lab7::arena::play_game;
use smallmobile_lab7::table::Table;
use smallmobile_lab8::agent::{Plan, Plans};
use smallmobile_lab8::encode::{GLOBAL, NODE};
use smallmobile_lab8::record::{self, EXTENSIONS, GLOBAL_MASK, LEGAL, MAP_COLS, PROBABILITY, SEG, STEP, half};

fn fixture_net() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/net.bin").to_string_lossy().into_owned()
}

/// Le décodeur d'un réseau au hasard, au hasard (t = 1) ou au plus probable, finit ses parties.
#[test]
fn the_decoder_plays_whole_games() {
    for t in ["1", "0"] {
        let plan = Plan::named(&format!("plan:net={},t={t}", fixture_net())).unwrap();
        for n in 2..=5 {
            let table = Table::new(n, 4000 + n as u64);
            let seats: Vec<&dyn Agent> = vec![&plan; n];
            let played = play_game(&table, n as u32, &seats, 9);
            assert_eq!(played.outcome.players, n);
        }
    }
}

/// La recherche sur des plans (réseau au hasard, 4 candidats, 2 mondes) finit ses parties.
#[test]
fn the_plan_search_plays_whole_games() {
    let plans = Plans::named(&format!("plans:net={},k=4,m=2,mods=2", fixture_net())).unwrap();
    let field = Plan::named(&format!("plan:net={},t=1", fixture_net())).unwrap();
    for n in 2..=5 {
        let table = Table::new(n, 5000 + n as u64);
        let mut seats: Vec<&dyn Agent> = vec![&field; n];
        seats[0] = &plans;
        let played = play_game(&table, n as u32, &seats, 11);
        assert_eq!(played.outcome.players, n);
    }
    use std::sync::atomic::Ordering::Relaxed;
    assert_eq!(smallmobile_lab8::agent::SHORT.load(Relaxed), 0, "plans trop courts");
    assert!(smallmobile_lab8::agent::RESUMED.load(Relaxed) > 20, "suites de plans reprises");
}

fn read(path: &Path, ext: &str) -> Vec<u8> {
    let mut name = path.as_os_str().to_owned();
    name.push(ext);
    std::fs::read(name).unwrap()
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

/// Un f16 en f32 (pour vérifier `record::half`).
fn unhalf(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (e, m) = ((h >> 10) & 0x1f, (h & 0x3ff) as f32);
    sign * match e {
        0 => m * 2f32.powi(-24),
        0x1f => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e as i32 - 15),
    }
}

#[test]
fn half_rounds_to_nearest() {
    for x in [0.0f32, 1.0, -2.5, 0.1, 1e-3, 3e-5, 1e-7, 65504.0, 0.333_333, 87.0, 1.0 - 1e-4] {
        let h = unhalf(half(x));
        let ulp = (x.abs() * 2f32.powi(-11)).max(2f32.powi(-25));
        assert!((h - x).abs() <= ulp, "{x} → {h}");
    }
    // l'égalité va au pair, et le report passe à l'exposant suivant
    assert_eq!(half(1.0 + 2f32.powi(-11)), 0x3c00);
    assert_eq!(half(2.0 - 2f32.powi(-12)), 0x4000);
}

/// Les parties d'imitation (c130 à n = 8) : la forme compacte redonne exactement les entrées pleines ;
/// chaque décision a ses coups possibles, le coup joué en est, la politique et l'a priori font 1 ; les
/// parts de victoire font 1 ; les tailles des fichiers se suivent.
#[test]
fn imitation_games_split_into_plans() {
    let c130 = Path::new(env!("CARGO_MANIFEST_DIR")).join("../lab7/nets/c130.bin");
    let player = agent::by_name(&format!("az:net={},n=8", c130.display())).unwrap();
    let dir = std::env::temp_dir().join(format!("lab8-record-{}", std::process::id()));
    let cfg = record::Config { players: vec![2, 5], games: 2, threads: 1, seed: 3, raw: true, concurrent: 1 };
    record::run(&cfg, player.az().unwrap(), &dir, None).unwrap();
    let prefix = dir.join("r3-00");
    let f: Vec<Vec<u8>> = EXTENSIONS.iter().map(|e| read(&prefix, e)).collect();
    let [games, map, adj, seg, nmask, nval, gmask, gval, owner, step, legal] = &f[..] else { panic!() };
    let raw = read(&prefix, ".raw");
    let games = String::from_utf8(games.clone()).unwrap();
    assert_eq!(seg.len() % SEG, 0);
    let plans = seg.len() / SEG;
    assert_eq!(games.lines().map(|l| l.split(' ').nth(4).unwrap().parse::<usize>().unwrap()).sum::<usize>(), plans);
    let (mut regions, mut steps, mut at, mut nv, mut gv, mut r_at) = (0, 0, 0, 0, 0, 0);
    let (mut game, mut map_at) = (u32::MAX, 0);
    let mut game_regions = 0;
    for (i, p) in seg.chunks(SEG).enumerate() {
        let g = u32::from_le_bytes(p[..4].try_into().unwrap());
        let (s, n_nz, g_nz) = (u16_at(p, 4) as usize, u16_at(p, 6) as usize, u16_at(p, 8) as usize);
        let (n, r) = (p[10] as usize, p[11] as usize);
        assert!(s >= 1 && r >= 1);
        if g != game {
            map_at += game_regions;
            game = g;
            game_regions = r;
        }
        assert_eq!(r, game_regions, "même carte pour toute la partie");
        // la forme compacte redonne les entrées pleines
        let plan_raw = &raw[r_at..r_at + r * NODE + GLOBAL + 30 * 4];
        r_at += plan_raw.len();
        let mut vals = nval[nv..nv + n_nz].iter();
        for row in 0..r {
            let full = &plan_raw[row * NODE..(row + 1) * NODE];
            assert_eq!(&map[(map_at + row) * MAP_COLS..(map_at + row + 1) * MAP_COLS], &full[..MAP_COLS]);
            let mask = u64::from_le_bytes(nmask[(regions + row) * 8..(regions + row + 1) * 8].try_into().unwrap());
            for (j, &x) in full[MAP_COLS..].iter().enumerate() {
                assert_eq!(if mask >> j & 1 == 1 { *vals.next().unwrap() } else { 0 }, x, "plan {i}, région {row}");
            }
        }
        assert!(vals.next().is_none());
        nv += n_nz;
        let mut vals = gval[gv..gv + g_nz].iter();
        for (j, &x) in plan_raw[r * NODE..r * NODE + GLOBAL].iter().enumerate() {
            let on = gmask[i * GLOBAL_MASK + j / 8] >> (j % 8) & 1 == 1;
            assert_eq!(if on { *vals.next().unwrap() } else { 0 }, x, "plan {i}, entrée {j} de la partie");
        }
        assert!(vals.next().is_none());
        gv += g_nz;
        let full: Vec<f32> =
            plan_raw[r * NODE + GLOBAL..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let v: Vec<f32> = (0..30).map(|k| unhalf(u16_at(p, 16 + 2 * k))).collect();
        for (k, (&a, &b)) in v.iter().zip(&full).enumerate() {
            assert_eq!(u16_at(p, 16 + 2 * k), half(b));
            assert!((a - b).abs() <= b.abs() * 1e-3 + 1e-4, "valeur {k} : {a} pour {b}");
        }
        for (what, from) in [("win", 0), ("here", 10), ("next_win", 15), ("prior_win", 25)] {
            let sum: f32 = v[from..from + n].iter().sum();
            assert!((sum - 1.0).abs() < 5e-3, "{what} : {sum}");
        }
        regions += r;
        for st in step[steps * STEP..(steps + s) * STEP].chunks(STEP) {
            let (key, count) = (u16_at(st, 0), u16_at(st, 2) as usize);
            assert!(count >= 2, "une décision a plusieurs coups possibles");
            let moves: Vec<(u16, u16, u16)> = legal[at * LEGAL..(at + count) * LEGAL]
                .chunks(LEGAL)
                .map(|c| (u16_at(c, 0), u16_at(c, 2), u16_at(c, 4)))
                .collect();
            assert!(moves.iter().any(|m| m.0 == key), "le coup joué est possible");
            for (what, total) in [
                ("politique", moves.iter().map(|m| m.1 as f32).sum::<f32>()),
                ("a priori", moves.iter().map(|m| m.2 as f32).sum()),
            ] {
                assert!((total - PROBABILITY).abs() <= count as f32, "{what} : {total}");
            }
            at += count;
        }
        steps += s;
    }
    assert_eq!(r_at, raw.len());
    assert_eq!((nv, gv), (nval.len(), gval.len()));
    assert_eq!(nmask.len(), regions * 8);
    assert_eq!(gmask.len(), plans * GLOBAL_MASK);
    assert_eq!(map.len(), (map_at + game_regions) * MAP_COLS);
    assert_eq!(adj.len(), (map_at + game_regions) * 8);
    assert_eq!(step.len(), steps * STEP);
    assert_eq!(legal.len(), at * LEGAL);
    assert_eq!(owner.len(), regions);
    assert!(plans > 50, "{plans} plans");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// La politique améliorée : l'a priori quand `beta` est nul, et après le dernier embranchement ; ailleurs,
/// elle penche vers les coups des meilleurs candidats.
#[test]
fn improved_policy_follows_the_best_candidates() {
    use smallmobile_lab8::net::Value;
    use smallmobile_lab8::search::{Decision, Searched, improved};
    let d = |keys: &[u16], probs: &[f32], chosen| Decision { keys: keys.to_vec(), probs: probs.to_vec(), chosen };
    // trois candidats : A = (1, 5), B = (1, 6), C = (2, 7) ; B est le meilleur
    let traces = vec![
        vec![d(&[1, 2, 3], &[0.5, 0.3, 0.2], 0), d(&[5, 6], &[0.6, 0.4], 0)],
        vec![d(&[1, 2, 3], &[0.5, 0.3, 0.2], 0), d(&[5, 6], &[0.6, 0.4], 1)],
        vec![d(&[1, 2, 3], &[0.5, 0.3, 0.2], 1), d(&[7, 8], &[0.9, 0.1], 0)],
    ];
    let value = Value { win: [0.2; 5], gain: [0.0; 5] };
    let s = Searched {
        plan: Vec::new(),
        candidates: vec![(Vec::new(), 0.4), (Vec::new(), 0.6), (Vec::new(), 0.3)],
        traces,
        root: (value, value),
        encodes: 0,
        steps: 0,
    };
    for chosen in 0..3 {
        for step in improved(&s, chosen, 0.0) {
            for (_, p, q) in step {
                assert!((p - q).abs() < 1e-6, "beta nul : l'a priori");
            }
        }
    }
    let t = improved(&s, 1, 10.0);
    let sum = |step: &[(u16, f32, f32)]| step.iter().map(|x| x.1).sum::<f32>();
    assert!((sum(&t[0]) - 1.0).abs() < 1e-5 && (sum(&t[1]) - 1.0).abs() < 1e-5);
    // premier pas : le coup 1 (A et B) gagne sur le coup 2 (C) ; le coup 3, sans candidat, reçoit la
    // moyenne de tous
    assert!(t[0][0].1 > 0.5 && t[0][1].1 < 0.3);
    // second pas : le coup 6 (B) l'emporte sur le coup 5 (A)
    assert!(t[1][1].1 > 0.4 && t[1][0].1 < 0.6);
    // le long de C, le second pas n'a qu'un candidat : l'a priori
    for (_, p, q) in &improved(&s, 2, 10.0)[1] {
        assert!((p - q).abs() < 1e-6);
    }
}

/// Le selfplay (réseau au hasard, sur le processeur) : chaque plan cherché est enregistré, ses
/// décisions ont leurs coups possibles, le coup joué en est, la cible et l'a priori font 1 ; Python
/// les lit comme les parties d'imitation.
#[test]
fn selfplay_games_split_into_plans() {
    use smallmobile_lab8::net::Net;
    use smallmobile_lab8::search::{self, Cpu};
    use smallmobile_lab8::selfplay;
    let net = Net::load(Path::new(&fixture_net())).unwrap();
    let dir = std::env::temp_dir().join(format!("lab8-selfplay-{}", std::process::id()));
    let cfg = selfplay::Config {
        players: vec![2, 4],
        games: 2,
        threads: 1,
        seed: 5,
        search: search::Config::parse("k=4,m=2,mods=2").unwrap(),
        beta: 10.0,
        explore: 1,
    };
    selfplay::run(&cfg, &Cpu(&net), &dir).unwrap();
    let prefix = dir.join("s5-00");
    let f: Vec<Vec<u8>> = EXTENSIONS.iter().map(|e| read(&prefix, e)).collect();
    let [games, _, _, seg, .., owner, step, legal] = &f[..] else { panic!() };
    let games = String::from_utf8(games.clone()).unwrap();
    let plans = seg.len() / SEG;
    assert_eq!(games.lines().map(|l| l.split(' ').nth(4).unwrap().parse::<usize>().unwrap()).sum::<usize>(), plans);
    let (mut steps, mut at, mut regions) = (0, 0, 0);
    for p in seg.chunks(SEG) {
        let (s, n) = (u16_at(p, 4) as usize, p[10] as usize);
        regions += p[11] as usize;
        assert!(s >= 1);
        let v: Vec<f32> = (0..30).map(|k| unhalf(u16_at(p, 16 + 2 * k))).collect();
        for (what, from) in [("win", 0), ("here", 10), ("next_win", 15), ("prior_win", 25)] {
            let sum: f32 = v[from..from + n].iter().sum();
            assert!((sum - 1.0).abs() < 5e-3, "{what} : {sum}");
        }
        for st in step[steps * STEP..(steps + s) * STEP].chunks(STEP) {
            let (key, count) = (u16_at(st, 0), u16_at(st, 2) as usize);
            assert!(count >= 2);
            let moves: Vec<(u16, u16, u16)> = legal[at * LEGAL..(at + count) * LEGAL]
                .chunks(LEGAL)
                .map(|c| (u16_at(c, 0), u16_at(c, 2), u16_at(c, 4)))
                .collect();
            assert!(moves.iter().any(|m| m.0 == key), "le coup joué est possible");
            for total in [moves.iter().map(|m| m.1 as f32).sum::<f32>(), moves.iter().map(|m| m.2 as f32).sum()] {
                assert!((total - PROBABILITY).abs() <= count as f32, "{total}");
            }
            at += count;
        }
        steps += s;
    }
    assert_eq!((step.len(), legal.len(), owner.len()), (steps * STEP, at * LEGAL, regions));
    assert!(plans > 40, "{plans} plans");
    std::fs::remove_dir_all(&dir).unwrap();
}
