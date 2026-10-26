//! Les shaders de l'encodeur sur GPU (`src/vulkan.rs`, option `vulkan`) : `shaders/*.comp` compilés en
//! SPIR-V par `glslc` (paquet `glslc` de Debian et Ubuntu, ou le SDK Vulkan).

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=shaders");
    if std::env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    for entry in std::fs::read_dir("shaders").expect("lab8/shaders") {
        let path = entry.expect("shader").path();
        if path.extension().is_none_or(|e| e != "comp") {
            continue;
        }
        let spv = Path::new(&out).join(path.with_extension("spv").file_name().expect("nom"));
        let status = Command::new("glslc")
            .args(["-O", "--target-env=vulkan1.3", "-I", "shaders"])
            .arg(&path)
            .arg("-o")
            .arg(&spv)
            .status()
            .unwrap_or_else(|e| panic!("glslc introuvable ({e}) : paquet glslc ou SDK Vulkan"));
        assert!(status.success(), "glslc : {}", path.display());
    }
}
