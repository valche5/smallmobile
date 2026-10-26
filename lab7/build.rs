//! Les shaders du réseau sur Vulkan (`src/vulkan.rs`, option `vulkan`) : `shaders/*.comp` compilés
//! en SPIR-V par `glslc` (paquet `glslc` de Debian et Ubuntu, ou le SDK Vulkan).

use std::path::Path;
use std::process::Command;

fn main() {
    source_stamp();
    println!("cargo::rerun-if-changed=shaders");
    if std::env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    for entry in std::fs::read_dir("shaders").expect("lab7/shaders") {
        let path = entry.expect("shader").path();
        if path.extension().is_none_or(|e| e != "comp") {
            continue;
        }
        let spv = Path::new(&out).join(path.with_extension("spv").file_name().expect("nom"));
        let status = Command::new("glslc")
            .args(["-O", "--target-env=vulkan1.3"])
            .arg(&path)
            .arg("-o")
            .arg(&spv)
            .status()
            .unwrap_or_else(|e| panic!("glslc introuvable ({e}) : paquet glslc ou SDK Vulkan"));
        assert!(status.success(), "glslc : {}", path.display());
    }
    // net.comp instrumenté : la durée de chaque étape (`LAB7_VULKAN_PHASES`, `src/vulkan.rs`)
    let status = Command::new("glslc")
        .args(["-O", "--target-env=vulkan1.3", "-DPROFILE", "shaders/net.comp", "-o"])
        .arg(Path::new(&out).join("net_prof.spv"))
        .status()
        .expect("glslc");
    assert!(status.success(), "glslc : net.comp (PROFILE)");
}

/// Empreinte du code qui produit les positions et les suites, indépendante du chemin de compilation.
fn source_stamp() {
    use std::io::Write;
    use std::process::Stdio;
    fn files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("sources") {
            let p = entry.expect("source").path();
            if p.is_dir() {
                files(&p, out);
            } else {
                out.push(p);
            }
        }
    }
    let mut paths = Vec::new();
    for dir in ["src", "brain/src", "shaders", "../kernel/src"] {
        println!("cargo::rerun-if-changed={dir}");
        files(Path::new(dir), &mut paths);
    }
    for file in ["build.rs", "Cargo.toml", "brain/Cargo.toml", "../kernel/Cargo.toml", "../Cargo.toml", "../Cargo.lock"]
    {
        println!("cargo::rerun-if-changed={file}");
        paths.push(file.into());
    }
    paths.sort();
    let mut hash = Command::new("timeout")
        .args(["15", "sha256sum"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("sha256sum");
    {
        let mut input = hash.stdin.take().expect("entrée du hachage");
        for p in paths {
            let bytes = std::fs::read(&p).expect("source lisible");
            writeln!(input, "{} {}", p.display(), bytes.len()).unwrap();
            input.write_all(&bytes).unwrap();
        }
    }
    let out = hash.wait_with_output().expect("empreinte des sources");
    assert!(out.status.success(), "sha256sum");
    let text = String::from_utf8(out.stdout).unwrap();
    println!("cargo::rustc-env=LAB7_SOURCE_SHA256={}", text.split_whitespace().next().unwrap());
}
