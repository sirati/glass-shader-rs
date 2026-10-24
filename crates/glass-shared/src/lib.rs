//! Parameters of the glass lens (a rounded slab of glass over a view or a
//! screen copy), as its host code writes them and its rust-gpu shaders
//! (`glass-shaders`) read them: `repr(C)`, 32-bit scalars, padded to 16
//! bytes.
#![cfg_attr(target_arch = "spirv", no_std)]

#[cfg(not(target_arch = "spirv"))]
use bytemuck::{Pod, Zeroable};

/// Glass geometry in draw pixels, with a transform into composition snapshots.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[cfg_attr(not(target_arch = "spirv"), derive(Pod, Zeroable))]
pub struct LensParams {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub radius: f32,
    pub rim: f32,
    pub glass: u32,
    pub bits: u32,
    pub snapshot_x: i32,
    pub snapshot_y: i32,
    pub snapshot_w: i32,
    pub snapshot_h: i32,
    pub quantized_float: u32,
    /// Window coordinates to snapshot texels, independent of pixel scale.
    pub sample_scale: f32,
    pub sample_x: f32,
    pub sample_y: f32,
    pub output_bits: u32,
    pub check_occlusion: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// The lens of one screen copy, drawn over the copy in the output pass:
/// the copy's rectangle of the source texture at (src_x, src_y) into the
/// target at (dst_x, dst_y), scaled by `zoom_num / zoom_den` by nearest
/// texel; `bits`: the quantized colour depth to expand from, 0 to read
/// values as they are.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[cfg_attr(not(target_arch = "spirv"), derive(Pod, Zeroable))]
pub struct CopyLensParams {
    pub src_x: i32,
    pub src_y: i32,
    pub dst_x: i32,
    pub dst_y: i32,
    pub zoom_num: u32,
    pub zoom_den: u32,
    pub bits: u32,
    /// 1: integer FP16 storage; 0: biased UNORM8 storage.
    pub quantized_float: u32,
    /// The lens, in final window pixels.
    pub lens_width: f32,
    pub lens_height: f32,
    pub lens_radius: f32,
    pub lens_rim: f32,
    pub lens_glass: u32,
    /// Original canvas presentation, for restoring corners and scattered light.
    pub canvas_x: i32,
    pub canvas_y: i32,
    pub canvas_num: u32,
    pub canvas_den: u32,
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub _pad: u32,
}
