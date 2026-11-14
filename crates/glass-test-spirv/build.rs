//! Compile this package's shader module using the shared host build helper.
#![forbid(unsafe_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    shader_build::build(shader_build::Config {
        shader_crate: std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?).join("../glass-test-shaders"),
        source_packages: &["glass-test-shaders", "glass-shaders", "glass-shared", "tile-render-shaders", "tile-render-shared"],
        precompiled_envs: &["GLASS_TEST_PRECOMPILED_SPV"],
        spirv_env: "GLASS_TEST_SPV",
        fallback_env: "GLASS_TEST_SPV_FALLBACK",
        output_name: "glass_test_shaders.spv",
        dual_source_entries: &["text_dual_fs", "subpixel_dual_fs"],
    })
}
