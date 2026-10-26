//! Les plans : le décodeur joue des parties entières sans coup impossible ; les parties d'imitation
//! se découpent en plans dont les cibles se tiennent.

use std::path::Path;

use smallmobile_lab7::agent::{self, Agent};
use smallmobile_lab7::arena::play_game;
use smallmobile_lab7::table::Table;
use smallmobile_lab8::agent::Plan;
use smallmobile_lab8::record::{self, LEGAL, PROBABILITY, SEG, STEP};

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

fn read(path: &Path, ext: &str) -> Vec<u8> {
    let mut name = path.as_os_str().to_owned();
    name.push(ext);
    std::fs::read(name).unwrap()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// Les parties d'imitation (c130 à n = 8) : chaque décision a ses coups possibles, le coup joué en est,
/// la politique fait 1 ; les parts de victoire font 1 ; les tailles des fichiers se suivent.
#[test]
fn imitation_games_split_into_plans() {
    let c130 = Path::new(env!("CARGO_MANIFEST_DIR")).join("../lab7/nets/c130.bin");
    let player = agent::by_name(&format!("az:net={},n=8", c130.display())).unwrap();
    let dir = std::env::temp_dir().join(format!("lab8-record-{}", std::process::id()));
    let cfg = record::Config { players: vec![2, 5], games: 2, threads: 1, seed: 3 };
    record::run(&cfg, player.az().unwrap(), &dir, None).unwrap();
    let prefix = dir.join("r3-00");
    let (seg, step, legal) = (read(&prefix, ".seg"), read(&prefix, ".step"), read(&prefix, ".legal"));
    let (node, glob, adj, owner) =
        (read(&prefix, ".node"), read(&prefix, ".glob"), read(&prefix, ".adj"), read(&prefix, ".owner"));
    let games = String::from_utf8(read(&prefix, ".games")).unwrap();
    assert_eq!(seg.len() % SEG, 0);
    let plans = seg.len() / SEG;
    assert_eq!(games.lines().map(|l| l.split(' ').nth(4).unwrap().parse::<usize>().unwrap()).sum::<usize>(), plans);
    let (mut regions, mut steps, mut at) = (0, 0, 0);
    for p in seg.chunks(SEG) {
        let (n, r, s) = (p[0] as usize, p[1] as usize, u16::from_le_bytes([p[2], p[3]]) as usize);
        assert!(s >= 1 && r >= 1);
        let v = f32s(&p[8..]);
        for (i, what) in ["win", "here", "next_win", "prior_win"].iter().enumerate() {
            let at = [0, 10, 15, 25][i];
            let sum: f32 = v[at..at + n].iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "{what} : {sum}");
        }
        regions += r;
        for st in step[steps * STEP..(steps + s) * STEP].chunks(STEP) {
            let key = u16::from_le_bytes([st[0], st[1]]);
            let count = u16::from_le_bytes([st[2], st[3]]) as usize;
            assert!(count >= 2, "une décision a plusieurs coups possibles");
            let word = |c: &[u8], i: usize| u16::from_le_bytes([c[2 * i], c[2 * i + 1]]);
            let moves: Vec<(u16, u16, u16)> = legal[at * LEGAL..(at + count) * LEGAL]
                .chunks(LEGAL)
                .map(|c| (word(c, 0), word(c, 1), word(c, 2)))
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
    assert_eq!(step.len(), steps * STEP);
    assert_eq!(legal.len(), at * LEGAL);
    assert_eq!(node.len(), regions * 77);
    assert_eq!(glob.len(), plans * 820);
    assert_eq!(adj.len(), regions * 8);
    assert_eq!(owner.len(), regions);
    assert!(plans > 50, "{plans} plans");
    std::fs::remove_dir_all(&dir).unwrap();
}
