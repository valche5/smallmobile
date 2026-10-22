//! Les shaders du réseau sur Vulkan (`src/vulkan.rs`, option `vulkan`) : `shaders/*.comp` compilés
//! en SPIR-V par `glslc` (paquet `glslc` de Debian et Ubuntu, ou le SDK Vulkan).

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=shaders");
    if std::env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    for entry in std::fs::read_dir("shaders").expect("lab3/shaders") {
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
    // net.comp instrumenté : la durée de chaque étape (`LAB3_VULKAN_PHASES`, `src/vulkan.rs`)
    let status = Command::new("glslc")
        .args(["-O", "--target-env=vulkan1.3", "-DPROFILE", "shaders/net.comp", "-o"])
        .arg(Path::new(&out).join("net_prof.spv"))
        .status()
        .expect("glslc");
    assert!(status.success(), "glslc : net.comp (PROFILE)");
}
