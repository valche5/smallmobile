//! Le réseau Rust (`net`) fait le même calcul que PyTorch (`lab8/py/model.py`) : sur 49 plans de parties
//! au hasard (`lab8 fixture --games 4 --seed 8 --every 6`), un réseau au hasard
//! (`check.py --random 32,2,4,1 --seed 1`, poids, sorties du décodeur comprises, tous non nuls).

use std::path::Path;

use smallmobile_lab8::net::Net;

#[test]
fn rust_and_pytorch_agree() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let net = Net::load(&dir.join("net.bin")).unwrap();
    let (heads, logits, plans) = smallmobile_lab8::fixture::check(&net, &dir).unwrap();
    assert_eq!(plans, 49);
    assert!(heads < 1e-4 && logits < 1e-4, "écarts {heads:e} et {logits:e}");
}
