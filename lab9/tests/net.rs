//! Le réseau Rust (`net`) fait le même calcul que PyTorch (`lab9/py/model.py`) : sur 49 plans de parties
//! au hasard (`lab9 fixture --games 4 --seed 8 --every 6`), deux réseaux au hasard (poids, sorties du
//! décodeur comprises, tous non nuls) : `tests/fixtures`, celui de lab8, tout transformer
//! (`check.py --random 32,2,4,1 --seed 1` de lab8, sans le tenseur `heads`), et `tests/fixtures-hybrid`,
//! hybride (`check.py --random 32,RTR,4,1 --seed 1`).

use std::path::Path;

use smallmobile_lab9::net::Net;

#[test]
fn rust_and_pytorch_agree() {
    for fixture in ["tests/fixtures", "tests/fixtures-hybrid"] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(fixture);
        let net = Net::load(&dir.join("net.bin")).unwrap();
        let (heads, logits, plans) = smallmobile_lab9::fixture::check(&net, &dir).unwrap();
        assert_eq!(plans, 49);
        assert!(heads < 1e-4 && logits < 1e-4, "{fixture} : écarts {heads:e} et {logits:e}");
    }
}

/// Les pas du décodeur menés ensemble (`step_batch`) donnent exactement les logits des pas seuls.
#[test]
fn batched_steps_match_lone_ones() {
    use smallmobile_kernel::Phase;
    use smallmobile_lab7::table::Table;
    use smallmobile_lab9::encode::encode;
    use smallmobile_lab9::know::Knowledge;
    use smallmobile_lab9::moves;
    use smallmobile_lab9::net::{Cache, StepIn};
    use smallmobile_lab9::rand::Rng;
    use smallmobile_lab9::tokens::{distances, dynamics};

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let net = Net::load(&dir.join("net.bin")).unwrap();
    let mut rng = Rng::new(4);
    let mut positions = Vec::new();
    for n in 2..=5 {
        let table = Table::new(n, 300 + n as u64);
        let dist = distances(&table.map);
        let mut g = table.game(n as u32);
        let mut k = Knowledge::default();
        while g.state.phase != Phase::GameOver && positions.len() < 6 * (n - 1) {
            let legal = moves::legal(&g, &k);
            if legal.len() > 1 && rng.below(5) == 0 {
                positions.push((
                    net.encode(&encode(&g, &k), &dist),
                    dynamics(&g, &k),
                    legal[rng.below(legal.len())].key,
                ));
            }
            k.play(&mut g, &legal[rng.below(legal.len())].mv).unwrap();
        }
    }
    // deux pas chacun : le premier, puis après un coup possible
    let mut lone = Vec::new();
    for (mem, dy, key) in &positions {
        let mut cache = Cache::default();
        lone.push(net.step(mem, &mut cache, None, dy));
        lone.push(net.step(mem, &mut cache, Some(*key), dy));
    }
    let mut caches: Vec<Cache> = positions.iter().map(|_| Cache::default()).collect();
    let mut together = Vec::new();
    for second in [false, true] {
        let mut items: Vec<StepIn> = positions
            .iter()
            .zip(caches.iter_mut())
            .map(|((mem, dy, key), cache)| StepIn { mem, cache, prev: second.then_some(*key), dy: *dy })
            .collect();
        together.push(net.step_batch(&mut items));
    }
    for (i, l) in lone.chunks(2).enumerate() {
        assert_eq!(l[0], together[0][i]);
        assert_eq!(l[1], together[1][i]);
    }
}
