use std::path::Path;
use std::sync::Arc;

use smallmobile_lab5::az::{Az, Config, Direct, Evals, now};
use smallmobile_lab5::know::Knowledge;
use smallmobile_lab5::moves;
use smallmobile_lab5::net::Net;
use smallmobile_lab5::probe::{self, Stage, Trial};
use smallmobile_lab5::rand::Rng;
use smallmobile_lab5::table::Table;

fn net(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("nets/{name}.bin"))
}
fn az(name: &str, n: usize) -> Az {
    Az::new(Config { iterations: n }, Arc::new(Net::load(&net(name)).unwrap()), name.into())
}

#[test]
fn the_brain_reproduces_lab3_with_c130() {
    use smallmobile_brain3 as old;
    let new = az("c130", 16);
    let old = old::search::Az::new(
        old::search::Config { iterations: 16 },
        Arc::new(old::net::Net::load(&net("c130")).unwrap()),
        "c130".into(),
    );
    for n in 2..=5 {
        let table = Table::new(n, 73 + n as u64);
        let (mut game, mut k, mut before) = (table.game(91), Knowledge::default(), old::know::Knowledge::default());
        let mut random = Rng::new(141);
        for step in 0..150 {
            if game.state.phase == smallmobile_kernel::Phase::GameOver {
                break;
            }
            let legal = moves::legal(&game, &k);
            if step % 29 == 0 && legal.len() > 1 {
                for explore in [false, true] {
                    let mut r1 = Rng::new(17);
                    let mut r2 = old::rand::Rng::new(17);
                    let a = new.search(&game, &k, &mut r1, explore, &mut Evals::default());
                    let b = old.search(&game, &before, &mut r2, explore, &mut old::search::Evals::default());
                    assert_eq!(format!("{:?}", a.mv), format!("{:?}", b.mv));
                    assert_eq!(a.policy, b.policy);
                    assert_eq!(a.visits, b.visits);
                    assert_eq!(a.finals, b.finals);
                    assert_eq!(a.wins, b.wins);
                    assert_eq!(a.value, b.value);
                    assert_eq!(r1.next_u64(), r2.next_u64());
                    for cap in [4, moves::KEYS] {
                        let keys = probe::candidates(&game, &legal, &a, cap, &mut Rng::new(51));
                        assert_eq!(keys.len(), cap.min(legal.len()));
                        assert_eq!(keys.iter().collect::<std::collections::HashSet<_>>().len(), keys.len());
                        for c in &legal {
                            if c.mv == a.mv || matches!(c.mv, moves::Move::Decline | moves::Move::EndConquest) {
                                assert!(keys.contains(&c.key));
                            }
                        }
                    }
                }
            }
            let choice = legal[random.below(legal.len())];
            let old_choice = old::moves::legal(&game, &before).into_iter().find(|c| c.key == choice.key).unwrap();
            before.play(&mut game.clone(), &old_choice.mv).unwrap();
            k.play(&mut game, &choice.mv).unwrap();
        }
    }
}

#[test]
fn selection_keeps_the_baseline_on_ties_and_confirmation_has_new_worlds() {
    let rows = vec![
        Trial { sample: 0, key: 1, world: 0, win: 0.5, margin: 0 },
        Trial { sample: 0, key: 2, world: 0, win: 0.5, margin: 0 },
    ];
    assert_eq!(probe::select(2, &[1, 2], &rows), 2);
    let worlds: std::collections::HashSet<_> = (0..100).map(|i| probe::seeds(19, 3, Stage::Select, i).0).collect();
    for i in 0..100 {
        assert!(!worlds.contains(&probe::seeds(19, 3, Stage::Confirm, i).0));
    }
}

#[test]
fn full_rollouts_ignore_the_original_hidden_world() {
    let az = az("gen0", 1);
    let table = Table::new(2, 25);
    let g = table.game(98);
    let k = Knowledge::default();
    let mut other = g;
    other.state.rng ^= 1381;
    other.state.race_deck.as_mut_slice().reverse();
    other.state.power_deck.as_mut_slice().reverse();
    let key = moves::legal(&g, &k)[0].key;
    let a = now(probe::rollout(&az, &g, &k, key, 551, 25, &Direct)).unwrap();
    let b = now(probe::rollout(&az, &other, &k, key, 551, 25, &Direct)).unwrap();
    assert_eq!(a.win, b.win);
    assert_eq!(a.coins, b.coins);
    assert_eq!(a.win.iter().sum::<f64>(), 1.0);
}

#[test]
fn three_stages_are_resumable_and_independent_of_cpu_scheduling() {
    let root = std::env::temp_dir().join(format!("lab5-probe-test-{}", std::process::id()));
    let az = az("gen0", 1);
    let mut cfg = probe::Config {
        players: vec![2],
        positions: 1,
        candidates: 4,
        select: 2,
        confirm: 2,
        seed: 39,
        threads: 1,
        concurrent: 1,
        out: root.join("a"),
    };
    // Le lanceur crée son journal avant le manifeste ; ce n'est pas un résultat orphelin.
    std::fs::create_dir_all(&cfg.out).unwrap();
    std::fs::write(cfg.out.join("positions.log"), "démarrage\n").unwrap();
    // Pas de campagne : trois petites parties de fixture, réseau gen0 et une simulation par choix.
    for stage in [Stage::Positions, Stage::Select, Stage::Confirm] {
        probe::run(&az, &cfg, stage, &net("gen0"), "cpu", None).unwrap();
    }
    let files: Vec<_> = std::fs::read_dir(&cfg.out)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .filter(|f| f.to_str().unwrap().starts_with('p') && f.to_str().unwrap().ends_with(".tsv"))
        .collect();
    cfg.threads = 2;
    cfg.concurrent = 3;
    // Une reprise vérifie les résultats présents sans les recalculer.
    for stage in [Stage::Positions, Stage::Select, Stage::Confirm] {
        probe::run(&az, &cfg, stage, &net("gen0"), "cpu", None).unwrap();
    }
    cfg.out = root.join("b");
    for stage in [Stage::Positions, Stage::Select, Stage::Confirm] {
        probe::run(&az, &cfg, stage, &net("gen0"), "cpu", None).unwrap();
    }
    for f in &files {
        assert_eq!(
            std::fs::read(root.join("a").join(f)).unwrap(),
            std::fs::read(root.join("b").join(f)).unwrap(),
            "{f:?}"
        );
    }
    // Le lecteur indépendant Python accepte le format réel et ses graines.
    let output = std::process::Command::new("timeout")
        .args(["30", "python3"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("py/probe.py"))
        .arg(&cfg.out)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(String::from_utf8_lossy(&output.stdout).contains("global équilibré"));
    cfg.seed += 1;
    assert!(probe::run(&az, &cfg, Stage::Select, &net("gen0"), "cpu", None).is_err());
    cfg.seed -= 1;
    let file = cfg.out.join("p000000.confirm.tsv");
    std::fs::write(&file, "# selected\t0\nsample\tkey\tworld\twin\tmargin\n").unwrap();
    assert!(probe::run(&az, &cfg, Stage::Confirm, &net("gen0"), "cpu", None).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
