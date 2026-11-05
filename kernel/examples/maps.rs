//! Réécrit `kernel/maps.bin`, les cartes du jeu (`smallmobile_kernel::maps`) :
//! `cargo run --release -p smallmobile-kernel --example maps`.

use smallmobile_kernel::maps;

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/maps.bin");
    let bytes = maps::build();
    std::fs::write(path, &bytes).expect("écriture de maps.bin");
    println!("{path} : {} cartes, {} octets", maps::PLAYERS.count() * maps::COUNT, bytes.len());
}
