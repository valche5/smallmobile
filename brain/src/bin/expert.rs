//! L'IA Expert pour le serveur du jeu : un processus natif, à côté du serveur Bun, qui répond aux
//! demandes de décision une à une (voir `src/server/expert.ts`).
//!
//! ```text
//! expert --net lab/nets/gen11.bin
//! ```
//!
//! Messages sur l'entrée et la sortie standard : un nombre de mots, puis les mots (32 bits,
//! petit-boutiste).
//! - demande : itérations, graine (2 mots), puis la partie (`wire::decode_game` : carte et état) ;
//! - réponse : 0, valeur estimée (f32), itérations, microsecondes de calcul, nombre d'actions, puis
//!   chaque action (son nombre de mots, puis `wire::encode_action`) ; ou un code négatif suivi du
//!   message d'erreur en UTF-8 (nombre d'octets, puis les octets rangés dans des mots).

use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use smallmobile_brain::expert::Expert;
use smallmobile_brain::net::Net;
use smallmobile_kernel::Game;
use smallmobile_kernel::wire::{self, Writer};

const REFUSED: i32 = -1;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let net = match args.as_slice() {
        [flag, path] if flag == "--net" => path.clone(),
        _ => {
            eprintln!("usage : expert --net poids.bin");
            std::process::exit(2)
        }
    };
    let net = Arc::new(Net::load(Path::new(&net)).unwrap_or_else(|e| {
        eprintln!("réseau illisible ({net}) : {e}");
        std::process::exit(2)
    }));
    let mut input = BufReader::new(io::stdin().lock());
    let mut output = BufWriter::new(io::stdout().lock());
    // une fin de l'entrée (serveur arrêté) termine le processus
    while let Some(request) = read_message(&mut input) {
        let reply = answer(&net, &request);
        if write_message(&mut output, &reply).is_err() {
            break;
        }
    }
}

fn answer(net: &Arc<Net>, request: &[u32]) -> Vec<u32> {
    match decide(net, request) {
        Ok(words) => words,
        Err(why) => {
            let mut out = vec![REFUSED as u32, why.len() as u32];
            out.extend(why.as_bytes().chunks(4).map(|c| {
                let mut b = [0u8; 4];
                b[..c.len()].copy_from_slice(c);
                u32::from_le_bytes(b)
            }));
            out
        }
    }
}

fn decide(net: &Arc<Net>, request: &[u32]) -> Result<Vec<u32>, String> {
    let [iterations, seed_lo, seed_hi, game @ ..] = request else {
        return Err("demande trop courte".into());
    };
    let (map, state) = wire::decode_game(game).map_err(|e| format!("partie illisible : {}", e.0))?;
    let start = Instant::now();
    let expert = Expert::new(net.clone(), (*iterations as usize).max(1));
    let seed = *seed_lo as u64 | (*seed_hi as u64) << 32;
    let d = expert.decide(&Game::new(&map, state), seed).map_err(|r| format!("aucun coup : {r:?}"))?;
    let micros = start.elapsed().as_micros().min(u32::MAX as u128) as u32;
    let mut out = vec![0, d.value.to_bits(), d.iterations as u32, micros, d.actions.len() as u32];
    let mut buffer = vec![0u32; 8 + map.len()];
    for a in &d.actions {
        let mut w = Writer::new(&mut buffer);
        wire::encode_action(a, map.len(), &mut w);
        out.push(w.len() as u32);
        out.extend_from_slice(w.written());
    }
    Ok(out)
}

fn read_message(r: &mut impl Read) -> Option<Vec<u32>> {
    let mut word = [0u8; 4];
    r.read_exact(&mut word).ok()?;
    let len = u32::from_le_bytes(word) as usize;
    let mut bytes = vec![0u8; len * 4];
    r.read_exact(&mut bytes).ok()?;
    Some(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn write_message(w: &mut impl Write, words: &[u32]) -> io::Result<()> {
    w.write_all(&(words.len() as u32).to_le_bytes())?;
    for x in words {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()
}
