//! Glass lens shaders: a rounded slab of glass (`lens_math`) over a view
//! drawn into a canvas (`lens_fs`: its rim, from snapshots of the canvas
//! before and after the view) and over a screen copy (`lens_copy_fs`, in
//! the output pass). Both draw a full-screen triangle (`lens_vs`) under a
//! scissor.
//!
//! The canvas is the tile renderer's: its quantized colour encoding
//! (`tile_render_shared`) is decoded on reading and encoded on writing.
#![cfg_attr(target_arch = "spirv", no_std)]

mod lens_math;

use glass_shared::{CopyLensParams, LensParams};
use spirv_std::glam::{IVec2, UVec2, Vec4};
use spirv_std::num_traits::Float;
use spirv_std::{Image, spirv};
use tile_render_shared::{QUANTIZED_BIAS, expand, quantize};

/// A triangle covering the viewport.
#[spirv(vertex)]
pub fn lens_vs(#[spirv(vertex_index)] vertex: u32, #[spirv(position)] out_pos: &mut Vec4) {
    let x = if vertex == 1 { 3.0 } else { -1.0 };
    let y = if vertex == 2 { 3.0 } else { -1.0 };
    *out_pos = Vec4::new(x, y, 0.0, 1.0);
}

/// A screen copy's lens over the copy just drawn (`params.src_*`,
/// `dst_*`, `zoom_*` as the copy's): its rim and halo, refracting the
/// copy's source; the clear face inside is left as the copy drew it.
#[spirv(fragment)]
pub fn lens_copy_fs(
    #[spirv(frag_coord)] frag: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] params: &CopyLensParams,
    #[spirv(descriptor_set = 0, binding = 0)] source: &Image!(2D, type = f32, sampled),
    output: &mut Vec4,
) {
    let p = LensParams { x: params.dst_x as f32, y: params.dst_y as f32,
        width: params.lens_width, height: params.lens_height, radius: params.lens_radius,
        rim: params.lens_rim, ..Default::default() };
    let s = lens_math::surface(&p, frag.x, frag.y);
    if s.distance < -p.rim.max(0.5) { spirv_std::arch::kill(); }
    let size = UVec2::new(params.canvas_width, params.canvas_height);
    let ratio = params.canvas_den as f32 / params.canvas_num as f32;
    let bounds = [0, 0, size.x as i32, size.y as i32];
    let zoom = params.zoom_den as f32 / params.zoom_num as f32;
    let sample = |channel: f32| {
        let bend = s.bend + s.dispersion * channel;
        // Preserve the clear face's nearest-pixel phase, then bend
        // in window pixels before converting to source texels.
        let x = params.src_x as f32 + ((frag.x - p.x) * zoom).floor() + 0.5 - s.nx * bend * zoom;
        let y = params.src_y as f32 + ((frag.y - p.y) * zoom).floor() + 0.5 - s.ny * bend * zoom;
        lens_sample(source, x, y, bounds, params.bits, params.quantized_float)
    };
    let base = lens_nearest(source, (frag.x - params.canvas_x as f32) * ratio,
            (frag.y - params.canvas_y as f32) * ratio,
            bounds, params.bits, params.quantized_float);
    let backdrop = if params.lens_glass != 0 { lens_math::halo(&s, base, p.rim) } else { base };
    let glass = if params.lens_glass != 0 {
        let (r, g, b) = (sample(1.0), sample(0.0), sample(-1.0));
        lens_math::shade(&s, lens_math::dispersed(r, g, b))
    } else { [if s.distance > -p.rim * 0.5 { 0.78 } else { 0.08 }; 3] };
    let rgb = lens_math::composite(&s, glass, backdrop);
    *output = Vec4::new(rgb[0], rgb[1], rgb[2], 1.0);
}

/// Restores cut-off corners and shades only the lens's narrow border.
#[spirv(fragment)]
pub fn lens_fs(
    #[spirv(frag_coord)] frag: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] p: &LensParams,
    #[spirv(descriptor_set = 0, binding = 0)] source: &Image!(2D, type = f32, sampled),
    #[spirv(descriptor_set = 0, binding = 2)] face: &Image!(2D, type = f32, sampled),
    #[spirv(descriptor_set = 0, binding = 3)] composed: &Image!(2D, type = f32, sampled),
    #[spirv(descriptor_set = 0, binding = 4)] shaded: &Image!(2D, type = f32, sampled),
    output: &mut Vec4,
) {
    let s = lens_math::surface(p, frag.x, frag.y);
    // Restore a source-pixel guard inside the bevel when re-shading boxed
    // pixels, so no old coarse highlight leaks onto the clear face.
    let guard = if p.sample_scale != 1.0 { 1.0 / p.sample_scale } else { 0.0 };
    if s.distance < -p.rim.max(0.5) - guard { spirv_std::arch::kill(); }
    let bounds = [p.snapshot_x, p.snapshot_y, p.snapshot_w, p.snapshot_h];
    let (x, y) = (frag.x * p.sample_scale + p.sample_x, frag.y * p.sample_scale + p.sample_y);
    if p.check_occlusion != 0 {
        // Compare stored texels exactly, before colour decoding. Later UI or
        // another camera can cover a rim, including with translucent art.
        let xy = IVec2::new(x.floor() as i32, y.floor() as i32);
        let before: Vec4 = shaded.fetch(IVec2::new((xy.x-p.snapshot_x).clamp(0,p.snapshot_w-1), (xy.y-p.snapshot_y).clamp(0,p.snapshot_h-1)));
        let after: Vec4 = composed.fetch(xy);
        if before.x != after.x || before.y != after.y || before.z != after.z { spirv_std::arch::kill(); }
    }
    // Backdrop art retains nearest boxed pixels, even when the rim is shaded
    // at final window resolution. Only the refracted face is interpolated.
    let base = lens_nearest(source, x, y, bounds, p.bits, p.quantized_float);
    let backdrop = if p.glass != 0 { lens_math::halo(&s, base, p.rim) } else { base };
    let rgb = if s.distance >= 0.5 {
        // Halo pixels have no glass coverage. Avoid three refracted samples
        // and the lighting calculation across the border's outer padding.
        backdrop
    } else {
        let glass = if s.distance < -p.rim.max(0.5) {
            lens_nearest(face, x, y, bounds, p.bits, p.quantized_float)
        } else if p.glass != 0 {
            let fetch = |channel: f32| {
                let bend = (s.bend + s.dispersion * channel) * p.sample_scale;
                lens_sample(face, x - s.nx * bend, y - s.ny * bend, bounds, p.bits, p.quantized_float)
            };
            let (r, g, b) = (fetch(1.0), fetch(0.0), fetch(-1.0));
            lens_math::shade(&s, lens_math::dispersed(r, g, b))
        } else { [if s.distance > -p.rim * 0.5 { 0.78 } else { 0.08 }; 3] };
        lens_math::composite(&s, glass, backdrop)
    };
    let encode = |v: f32| if p.output_bits == 0 { v } else {
        let q = quantize((v * 255.0 + 0.5) as u32, p.output_bits);
        if p.quantized_float != 0 { (q + tile_render_shared::FLOAT_QUANTIZED_BIAS) as f32 }
        else { (q + QUANTIZED_BIAS) as f32 / 255.0 }
    };
    *output = Vec4::new(encode(rgb[0]), encode(rgb[1]), encode(rgb[2]), 1.0);
}

/// Backdrop and clear-face guards keep boxed art. Fetch a single texel rather
/// than running four-tap interpolation with three zero weights on every rim pixel.
fn lens_nearest(source: &Image!(2D, type = f32, sampled), x: f32, y: f32,
    bounds: [i32; 4], bits: u32, float_encoding: u32) -> [f32; 3] {
    let v: Vec4 = source.fetch(IVec2::new(
        (x.floor() as i32 - bounds[0]).clamp(0, bounds[2]-1),
        (y.floor() as i32 - bounds[1]).clamp(0, bounds[3]-1)));
    if bits == 0 { [v.x, v.y, v.z] }
    else { [to_unorm(v.x, bits, float_encoding), to_unorm(v.y, bits, float_encoding), to_unorm(v.z, bits, float_encoding)] }
}

/// Bilinear refraction avoids bright nearest-sample ticks along straight edges.
/// Decode each quantized texel before interpolation, never interpolate its code.
fn lens_sample(source: &Image!(2D, type = f32, sampled), x: f32, y: f32,
    bounds: [i32; 4], bits: u32, float_encoding: u32) -> [f32; 3] {
    let (x, y) = (x - 0.5 - bounds[0] as f32, y - 0.5 - bounds[1] as f32);
    let (ix, iy) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - x.floor(), y - y.floor());
    let fetch = |dx: i32, dy: i32| {
        let v: Vec4 = source.fetch(IVec2::new((ix+dx).clamp(0, bounds[2]-1), (iy+dy).clamp(0, bounds[3]-1)));
        if bits == 0 { v } else { Vec4::new(to_unorm(v.x, bits, float_encoding), to_unorm(v.y, bits, float_encoding), to_unorm(v.z, bits, float_encoding), 1.0) }
    };
    let top = fetch(0,0) * (1.0-fx) + fetch(1,0) * fx;
    let bottom = fetch(0,1) * (1.0-fx) + fetch(1,1) * fx;
    let v = top * (1.0-fy) + bottom * fy;
    [v.x, v.y, v.z]
}


/// A stored canvas value as 8-bit colour (0..1): quantized ones expanded.
fn to_unorm(stored: f32, bits: u32, float_encoding: u32) -> f32 {
    let s = if float_encoding != 0 { stored as u32 } else { to_u8(stored) };
    let bias = if float_encoding != 0 { tile_render_shared::FLOAT_QUANTIZED_BIAS } else { QUANTIZED_BIAS };
    let q = if s > bias { s - bias } else { 0 };
    expand(q, bits) as f32 / 255.0
}

fn to_u8(v: f32) -> u32 {
    round(v * 255.0)
}

/// `v` (in `[0, 2^23)`) rounded to the nearest whole number (half to even),
/// from the bits of `v + 2^23`: no saturating conversion.
fn round(v: f32) -> u32 {
    (v + 8_388_608.0).to_bits() - 0x4B00_0000
}
