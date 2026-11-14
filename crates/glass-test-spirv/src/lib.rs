//! Test shader bytes compiled independently of the viewer application.
#![forbid(unsafe_code)]
pub const SPIRV: &[u8] = include_bytes!(env!("GLASS_TEST_SPV"));
pub const SPIRV_FALLBACK: &[u8] = include_bytes!(env!("GLASS_TEST_SPV_FALLBACK"));
