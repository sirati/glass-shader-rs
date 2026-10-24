//! The lens's CPU oracle: what its GPU passes paint, pixel for pixel
//! (`tile_render::reference::EffectOracle`).
use glass_shared::LensParams;
use tile_render::reference::EffectOracle;
use tile_render::{Camera, ScreenCopy};

use crate::{Lens, lens_math};

/// The lens's CPU oracle.
#[derive(Copy, Clone, Debug, Default)]
pub struct LensOracle;

/// See [`LensOracle`].
pub const ORACLE: LensOracle = LensOracle;

impl EffectOracle for LensOracle {
    fn view(&self, camera: &Camera, size: (u32, u32), backdrop: &[[f32; 3]], face: &[[f32; 3]]) -> Vec<(usize, [f32; 3])> {
        paint(backdrop, face, size, camera, None)
    }

    fn copy(&self, copy: &ScreenCopy, size: (u32, u32), picture: &[[f32; 3]]) -> Vec<(usize, [f32; 3])> {
        let (n, d) = (copy.zoom.0.max(1), copy.zoom.1.max(1));
        let shown = ((copy.source[2] as u32 * n).div_ceil(d), (copy.source[3] as u32 * n).div_ceil(d));
        let camera = Camera { origin: copy.at, effect: copy.effect, ..Camera::new(shown) };
        paint(picture, picture, size, &camera, Some(copy))
    }
}

/// The lens of `camera` (or of `copy`, `camera` its rectangle) over a
/// canvas of size (w, h): the pixels it paints. Its backdrop is
/// `backdrop`; it refracts `face` (a copy's: its source in `backdrop`).
fn paint(backdrop: &[[f32; 3]], face: &[[f32; 3]], (w, h): (u32, u32), camera: &Camera, copy: Option<&ScreenCopy>) -> Vec<(usize, [f32; 3])> {
    let lens = Lens::of(camera.effect.expect("a camera with a lens"));
    let mut out = Vec::new();
    let p = LensParams { x: camera.origin.0 as f32, y: camera.origin.1 as f32, width: camera.viewport.0 as f32, height: camera.viewport.1 as f32,
        radius: lens.radius, rim: lens.rim, glass: u32::from(lens.glass), ..Default::default() };
    let halo = lens.halo();
    let clip = camera.target_clip.unwrap_or([0, 0, w as i32, h as i32]);
    for y in (camera.origin.1-halo).max(clip[1]).max(0)..(camera.origin.1+camera.viewport.1 as i32+halo).min(clip[3]).min(h as i32) {
      for x in (camera.origin.0-halo).max(clip[0]).max(0)..(camera.origin.0+camera.viewport.0 as i32+halo).min(clip[2]).min(w as i32) {
        let s = lens_math::surface(&p, x as f32 + 0.5, y as f32 + 0.5);
        let i = (y*w as i32+x) as usize;
        let sample = |channel: f32| -> [f32; 3] {
            let bend = s.bend + s.dispersion * channel;
            let (px, py, image) = if let Some(copy) = copy {
                let ratio = copy.zoom.1.max(1) as f32 / copy.zoom.0.max(1) as f32;
                (copy.source[0] as f32 + (((x-copy.at.0) as f32+0.5)*ratio).floor() - s.nx*bend*ratio,
                 copy.source[1] as f32 + (((y-copy.at.1) as f32+0.5)*ratio).floor() - s.ny*bend*ratio, backdrop)
            } else { (x as f32 - s.nx*bend, y as f32 - s.ny*bend, face) };
            let (ix, iy) = (px.floor() as i32, py.floor() as i32);
            let (fx, fy) = (px-px.floor(), py-py.floor());
            let fetch = |dx: i32, dy: i32| image[(
                (iy+dy).clamp(0, h as i32-1)*w as i32+(ix+dx).clamp(0, w as i32-1)) as usize];
            let (a,b,c,d) = (fetch(0,0),fetch(1,0),fetch(0,1),fetch(1,1));
            std::array::from_fn(|ch| (a[ch]*(1.0-fx)+b[ch]*fx)*(1.0-fy)+(c[ch]*(1.0-fx)+d[ch]*fx)*fy)
        };
        if s.distance >= -p.rim.max(0.5) {
            let base = backdrop[i];
            let background = if lens.glass { lens_math::halo(&s, base, lens.rim) } else { base };
            let glass = if lens.glass {
                lens_math::shade(&s, lens_math::dispersed(sample(1.0), sample(0.0), sample(-1.0)))
            } else { [if s.distance > -p.rim*0.5 { 0.78 } else { 0.08 }; 3] };
            out.push((i, lens_math::composite(&s, glass, background)));
        }
      }
    }
    out
}
