//! A glass lens over a tile renderer's views and screen copies: a
//! rounded slab of glass (`lens_math`, shared with its shaders in
//! `glass-shaders`) whose rim refracts what is under it, with a halo
//! around it, or a plain rounded frame. It is the renderer's view effect
//! (`tile_render::Effect`): [`effect`] makes its GPU passes, [`ORACLE`] is
//! their CPU oracle.
#![forbid(unsafe_code)]

mod gpu;
mod oracle;

#[path = "../../glass-shaders/src/lens_math.rs"]
mod lens_math;

pub use oracle::{LensOracle, ORACLE};

/// A lens over a view or a screen copy: its rounded corners' `radius` and
/// its `rim`'s width (composition pixels; `Camera::boxed` scales these
/// along with the view so their window size is unchanged), polished glass
/// or a plain frame.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Lens {
    pub radius: f32,
    pub rim: f32,
    pub glass: bool,
}

impl Lens {
    /// Signed distance to the rounded rectangle; negative inside.
    pub fn distance(self, size: (u32, u32), p: (f32, f32)) -> f32 {
        let (w, h) = (size.0 as f32, size.1 as f32);
        let r = self.radius.clamp(0.0, w.min(h) * 0.5);
        let (x, y) = ((p.0 - w * 0.5).abs() - w * 0.5 + r, (p.1 - h * 0.5).abs() - h * 0.5 + r);
        (x.max(0.0) * x.max(0.0) + y.max(0.0) * y.max(0.0)).sqrt() + x.max(y).min(0.0) - r
    }

    /// The lens of a view or copy (its effect is one).
    pub fn of(effect: tile_render::Effect) -> Lens {
        let [radius, rim] = effect.lengths;
        Lens { radius, rim, glass: effect.variant != 0 }
    }

    /// How far the glass's halo reaches beyond the lens's rectangle.
    fn halo(self) -> i32 {
        if self.glass { (self.rim * 0.28).ceil() as i32 } else { 0 }
    }
}

impl From<Lens> for tile_render::Effect {
    fn from(lens: Lens) -> Self {
        tile_render::Effect { lengths: [lens.radius, lens.rim], variant: u32::from(lens.glass) }
    }
}

/// The lens's GPU passes, from a shader module holding `glass-shaders`'
/// entry points (`tile_render::Shaders::effect`).
pub fn effect(device: &wgpu::Device, shader: &wgpu::ShaderModule) -> Box<dyn tile_render::effect::ViewEffect> {
    Box::new(gpu::State::new(device, shader))
}
