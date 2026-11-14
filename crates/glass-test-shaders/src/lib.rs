//! Library GPU-test entry points. No application shaders are linked.
#![cfg_attr(target_arch = "spirv", no_std)]

pub use tile_render_shaders;
pub use glass_shaders;
