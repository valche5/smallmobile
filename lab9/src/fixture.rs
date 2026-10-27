//! La comparaison du réseau Rust à PyTorch, sur les plans écrits par `lab9 fixture` (`fixture.in`)
//! et les sorties qu'en tire `lab9/py/check.py` (`fixture.out` : par plan, les 20 têtes, puis 608
//! logits par pas, en f32).

use std::path::Path;

use crate::encode::{Encoded, GLOBAL, NODE, REGIONS};
use crate::net::{Cache, KEYS, NEG, Net};
use crate::tokens::DYN;

/// Écarts maximaux des têtes et des logits (coups sur une région présente, coups globaux), et nombre
/// de plans.
pub fn check(net: &Net, dir: &Path) -> std::io::Result<(f32, f32, usize)> {
    let inp = std::fs::read(dir.join("fixture.in"))?;
    let out: Vec<f32> =
        std::fs::read(dir.join("fixture.out"))?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    let (mut at, mut o) = (0, 0);
    let (mut heads, mut logits, mut plans) = (0f32, 0f32, 0);
    while at < inp.len() {
        let mut e = Encoded {
            regions: inp[at + 1] as usize,
            players: inp[at] as usize,
            node: [[0; NODE]; REGIONS],
            global: [0; GLOBAL],
        };
        at += 2;
        for row in e.node.iter_mut() {
            row.copy_from_slice(&inp[at..at + NODE]);
            at += NODE;
        }
        e.global.copy_from_slice(&inp[at..at + GLOBAL]);
        at += GLOBAL;
        let dist = &inp[at..at + REGIONS * REGIONS];
        at += REGIONS * REGIONS;
        let steps = inp[at] as usize;
        at += 1;
        let mem = net.encode(&e, dist);
        for (a, b) in mem.heads.iter().zip(&out[o..o + 20]) {
            heads = heads.max((a - b).abs());
        }
        o += 20;
        let mut cache = Cache::default();
        for _ in 0..steps {
            let prev = u16::from_le_bytes([inp[at], inp[at + 1]]);
            let dy: [u8; DYN] = inp[at + 2..at + 2 + DYN].try_into().unwrap();
            at += 2 + DYN;
            let got = net.step(&mem, &mut cache, (prev != u16::MAX).then_some(prev), &dy);
            for (a, b) in got.iter().zip(&out[o..o + KEYS]) {
                if *a > NEG / 2.0 {
                    logits = logits.max((a - b).abs());
                }
            }
            o += KEYS;
        }
        plans += 1;
    }
    assert_eq!(o, out.len(), "fixture.out ne suit pas fixture.in");
    Ok((heads, logits, plans))
}
