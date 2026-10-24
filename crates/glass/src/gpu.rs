//! Bounded backdrop copies and edge draws for magnifier cameras, and the
//! lens of a screen copy in the output pass.
use std::cell::RefCell;
use std::num::NonZeroU64;

use glass_shared::{CopyLensParams, LensParams};
use tile_render::effect::{CONTROL_SLOTS, CopyPlacement, EffectContext, ViewEffect};
use tile_render::renderer::MAX_COPIES;
use tile_render::{Camera, MAX_VIEWS, Present, ScreenCopy};

use crate::Lens;

/// Parameters of each draw in one buffer at this stride (the minimum
/// uniform offset alignment wgpu guarantees).
const PARAMS_STRIDE: u64 = 256;

pub(crate) struct State {
    layout: wgpu::BindGroupLayout,
    copy_layout: wgpu::BindGroupLayout,
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    copy_pipeline_layout: wgpu::PipelineLayout,
    /// The rim pipeline for each format it has been drawn into: the
    /// internal canvas (`Rgba8Unorm`, `Rgba16Float`), the caller's target
    /// when drawn directly, and the window a scaled picture is shown on
    /// (`Bgra8Unorm` or `Rgba8Unorm`).
    pipelines: RefCell<Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)>>,
    /// The screen copies' lens pipeline for each target format.
    copy_pipelines: RefCell<Vec<(wgpu::TextureFormat, wgpu::RenderPipeline)>>,
    params: [wgpu::Buffer; CONTROL_SLOTS],
    copy_params: [wgpu::Buffer; CONTROL_SLOTS],
    // Before-camera backdrop, clear face, then shaded rim. The last image
    // lets the window pass detect pixels covered by subsequent interface art.
    snapshots: Vec<Option<[wgpu::Texture; 3]>>,
    active: Vec<(usize, Camera, LensParams)>,
    /// The view being drawn, snapshotted before it.
    drawing: Option<(usize, LensParams)>,
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, size: usize) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding, visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: NonZeroU64::new(size as u64) }, count: None,
    }
}

fn begin_pass<'e>(encoder: &'e mut wgpu::CommandEncoder, view: &wgpu::TextureView, timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("glass lens"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None,
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

impl State {
    pub fn new(device: &wgpu::Device, shader: &wgpu::ShaderModule) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lens backdrop"),
            entries: &[texture_entry(0), texture_entry(2), texture_entry(3), texture_entry(4), uniform_entry(1, size_of::<LensParams>())],
        });
        let copy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("screen copy lens"),
            entries: &[texture_entry(0), uniform_entry(1, size_of::<CopyLensParams>())],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("lens"), bind_group_layouts: &[Some(&layout)], ..Default::default() });
        let copy_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("screen copy lens"), bind_group_layouts: &[Some(&copy_layout)], ..Default::default() });
        let params = std::array::from_fn(|_| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lens parameters"), size: PARAMS_STRIDE * (MAX_VIEWS * 2) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        }));
        let copy_params = std::array::from_fn(|_| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screen copy lens parameters"), size: PARAMS_STRIDE * MAX_COPIES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        }));
        Self { layout, copy_layout, shader: shader.clone(), pipeline_layout, copy_pipeline_layout, pipelines: RefCell::new(Vec::new()),
            copy_pipelines: RefCell::new(Vec::new()), params, copy_params, snapshots: Vec::new(), active: Vec::new(), drawing: None }
    }

    /// The pipeline running `fragment` drawing into `format`, made when first needed.
    fn pipeline(&self, device: &wgpu::Device, format: wgpu::TextureFormat, copy: bool) -> wgpu::RenderPipeline {
        let (pipelines, layout, fragment, label) = if copy {
            (&self.copy_pipelines, &self.copy_pipeline_layout, "lens_copy_fs", "screen copy lens")
        } else {
            (&self.pipelines, &self.pipeline_layout, "lens_fs", "polished lens rim")
        };
        if let Some((_, p)) = pipelines.borrow().iter().find(|(f, _)| *f == format) { return p.clone(); }
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{label} {format:?}")), layout: Some(layout),
            vertex: wgpu::VertexState { module: &self.shader, entry_point: Some("lens_vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &self.shader, entry_point: Some(fragment), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview_mask: None, cache: None,
        });
        pipelines.borrow_mut().push((format, pipeline.clone()));
        pipeline
    }

    fn snapshot(&mut self, cx: &EffectContext<'_>, encoder: &mut wgpu::CommandEncoder, canvas: &wgpu::Texture, camera: &Camera, view: usize) -> Option<LensParams> {
        let lens = Lens::of(camera.effect?);
        let pad = (lens.rim * 0.28).ceil() as i32 + 1;
        let (x, y) = ((camera.origin.0 - pad).max(0), (camera.origin.1 - pad).max(0));
        let (r, b) = ((camera.origin.0 + camera.viewport.0 as i32 + pad).min(canvas.width() as i32), (camera.origin.1 + camera.viewport.1 as i32 + pad).min(canvas.height() as i32));
        if r <= x || b <= y { return None; }
        let size = wgpu::Extent3d { width: (r - x) as u32, height: (b - y) as u32, depth_or_array_layers: 1 };
        self.snapshots.resize_with(self.snapshots.len().max(view + 1), || None);
        if self.snapshots[view].as_ref().is_none_or(|t| t[0].width() < size.width || t[0].height() < size.height || t[0].format() != canvas.format()) {
            self.snapshots[view] = Some(std::array::from_fn(|_| cx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("bounded lens backdrop"), size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: canvas.format(),
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
            })));
        }
        let source = wgpu::TexelCopyTextureInfo { origin: wgpu::Origin3d { x: x as u32, y: y as u32, z: 0 }, ..canvas.as_image_copy() };
        encoder.copy_texture_to_texture(source, self.snapshots[view].as_ref().unwrap()[0].as_image_copy(), size);
        let bits = cx.bits;
        let params = LensParams { x: camera.origin.0 as f32, y: camera.origin.1 as f32, width: camera.viewport.0 as f32, height: camera.viewport.1 as f32,
            radius: lens.radius, rim: lens.rim, glass: u32::from(lens.glass), bits, snapshot_x: x, snapshot_y: y, snapshot_w: r - x, snapshot_h: b - y, quantized_float: cx.quantized_float,
            sample_scale: 1.0, output_bits: bits, ..Default::default() };
        cx.queue.write_buffer(&self.params[cx.slot], view as u64 * PARAMS_STRIDE, bytemuck::bytes_of(&params));
        self.active.push((view, *camera, params));
        Some(params)
    }

    fn capture(&self, encoder: &mut wgpu::CommandEncoder, canvas: &wgpu::Texture, p: LensParams, view: usize, shaded: bool) {
        let source = wgpu::TexelCopyTextureInfo { origin: wgpu::Origin3d { x: p.snapshot_x as u32, y: p.snapshot_y as u32, z: 0 }, ..canvas.as_image_copy() };
        encoder.copy_texture_to_texture(source, self.snapshots[view].as_ref().unwrap()[if shaded { 2 } else { 1 }].as_image_copy(),
            wgpu::Extent3d { width: p.snapshot_w as u32, height: p.snapshot_h as u32, depth_or_array_layers: 1 });
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_with(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::RenderPipeline, canvas: &wgpu::TextureView, camera: &Camera, p: LensParams, slot: usize, view: usize, composed: Option<&wgpu::Texture>, timing: Option<wgpu::RenderPassTimestampWrites<'_>>) {
        let presented = composed.is_some();
        let snapshots = self.snapshots[view].as_ref().unwrap();
        let backdrop = snapshots[0].create_view(&Default::default());
        let face = snapshots[1].create_view(&Default::default());
        let shaded = snapshots[2].create_view(&Default::default());
        let composed = composed.unwrap_or(&snapshots[0]).create_view(&Default::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("lens backdrop"), layout: &self.layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&backdrop) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&face) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&composed) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&shaded) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.params[slot], offset: 0, size: NonZeroU64::new(size_of::<LensParams>() as u64) }) },
        ] });
        let mut pass = begin_pass(encoder, canvas, timing);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[((view + if presented { MAX_VIEWS } else { 0 }) as u64 * PARAMS_STRIDE) as u32]);
        let sx = |x: i32| ((x as f32 - p.sample_x) / p.sample_scale).floor() as i32;
        let sy = |y: i32| ((y as f32 - p.sample_y) / p.sample_scale).floor() as i32;
        let bounds = [sx(p.snapshot_x), sy(p.snapshot_y), sx(p.snapshot_x + p.snapshot_w), sy(p.snapshot_y + p.snapshot_h)];
        let clip = camera.target_clip.unwrap_or(bounds);
        let guard = if presented { (1.0 / p.sample_scale).ceil() as i32 } else { 0 };
        for [x, y, w, h] in edge_rects(camera, guard) {
            let (l, t) = (x.max(clip[0]).max(bounds[0]), y.max(clip[1]).max(bounds[1]));
            let (r, b) = ((x + w).min(clip[2]).min(bounds[2]), (y + h).min(clip[3]).min(bounds[3]));
            if r > l && b > t { pass.set_scissor_rect(l as u32, t as u32, (r - l) as u32, (b - t) as u32); pass.draw(0..3, 0..1); }
        }
    }
}

impl ViewEffect for State {
    fn bytes(&self) -> u64 {
        self.params.iter().chain(&self.copy_params).map(|p| p.size()).sum::<u64>() + self.snapshots.iter().flatten().flatten().map(|t|
            u64::from(t.width()) * u64::from(t.height()) * if t.format() == wgpu::TextureFormat::Rgba16Float { 8 } else { 4 }).sum::<u64>()
    }

    fn begin_frame(&mut self, keep_before: usize) {
        self.active.retain(|(view, _, _)| *view < keep_before);
    }

    /// Snapshots only this lens's bounds before its camera draws.
    fn before_view(&mut self, cx: &EffectContext<'_>, encoder: &mut wgpu::CommandEncoder, canvas: &wgpu::Texture, camera: &Camera, view: usize) {
        self.drawing = self.snapshot(cx, encoder, canvas, camera, view).map(|p| (view, p));
    }

    /// Draws the rim of the view into the canvas, between snapshots of its
    /// clear face and of the shaded result.
    fn after_view(&mut self, cx: &EffectContext<'_>, encoder: &mut wgpu::CommandEncoder, canvas: &wgpu::Texture, camera: &Camera, view: usize) {
        let Some((v, params)) = self.drawing.take().filter(|(v, _)| *v == view) else { return };
        self.capture(encoder, canvas, params, v, false);
        let pipeline = self.pipeline(cx.device, canvas.format(), false);
        let target = canvas.create_view(&Default::default());
        self.draw_with(cx.device, encoder, &pipeline, &target, camera, params, cx.slot, v, None, cx.timed(|| format!("v{v} lens rim")));
        self.capture(encoder, canvas, params, v, true);
    }

    /// Re-shade at physical window resolution. The art remains boxed, but the
    /// glass, dispersion and silhouette coverage do not become 2x2 blocks.
    fn presented(&self, cx: &EffectContext<'_>, encoder: &mut wgpu::CommandEncoder, target: &wgpu::Texture, composed: &wgpu::Texture, present: Present) {
        let scale = present.zoom.0.max(1) as f32 / present.zoom.1.max(1) as f32;
        if scale == 1.0 && present.at == (0, 0) { return; }
        let canvas = target.create_view(&Default::default());
        let pipeline = self.pipeline(cx.device, target.format(), false);
        for &(view, camera, mut p) in &self.active {
            p.x = p.x * scale + present.at.0 as f32;
            p.y = p.y * scale + present.at.1 as f32;
            p.width *= scale; p.height *= scale; p.radius *= scale; p.rim *= scale;
            p.sample_scale = 1.0 / scale;
            p.sample_x = -present.at.0 as f32 / scale;
            p.sample_y = -present.at.1 as f32 / scale;
            p.output_bits = 0;
            p.check_occlusion = 1;
            let transform = |v: i32, at: i32| (v as f32 * scale).floor() as i32 + at;
            let clip = camera.target_clip.unwrap_or([0, 0, present.canvas.0 as i32, present.canvas.1 as i32]);
            let camera = Camera { origin: (p.x.floor() as i32, p.y.floor() as i32), viewport: (p.width.ceil() as u32, p.height.ceil() as u32),
                effect: Some(Lens { radius: p.radius, rim: p.rim, glass: p.glass != 0 }.into()),
                target_clip: Some([transform(clip[0], present.at.0).max(0), transform(clip[1], present.at.1).max(0),
                    transform(clip[2], present.at.0).min(target.width() as i32), transform(clip[3], present.at.1).min(target.height() as i32)]), ..camera };
            cx.queue.write_buffer(&self.params[cx.slot], (MAX_VIEWS + view) as u64 * PARAMS_STRIDE, bytemuck::bytes_of(&p));
            self.draw_with(cx.device, encoder, &pipeline, &canvas, &camera, p, cx.slot, view, Some(composed), cx.timed(|| format!("v{view} window lens rim")));
        }
    }

    /// The copy's rim and halo, in final window pixels, over the copy.
    fn over_copy(&self, cx: &EffectContext<'_>, pass: &mut wgpu::RenderPass<'_>, copy: &ScreenCopy, placement: CopyPlacement, source: &wgpu::Texture, present: Present) {
        let Some(effect) = copy.effect else { return };
        let lens = Lens::of(effect);
        let (n, d) = (present.zoom.0.max(1), present.zoom.1.max(1));
        let factor = n as f32 / d as f32;
        let mut params = CopyLensParams { src_x: placement.src.0, src_y: placement.src.1, dst_x: placement.dst.0, dst_y: placement.dst.1,
            zoom_num: placement.zoom.0, zoom_den: placement.zoom.1, bits: cx.bits, quantized_float: cx.quantized_float, ..Default::default() };
        params.lens_width = (copy.source[2] as u32 * copy.zoom.0).div_ceil(copy.zoom.1) as f32 * factor;
        params.lens_height = (copy.source[3] as u32 * copy.zoom.0).div_ceil(copy.zoom.1) as f32 * factor;
        params.lens_radius = lens.radius * factor;
        params.lens_rim = lens.rim * factor;
        params.lens_glass = u32::from(lens.glass);
        params.canvas_x = present.at.0;
        params.canvas_y = present.at.1;
        params.canvas_num = n;
        params.canvas_den = d;
        params.canvas_width = source.width();
        params.canvas_height = source.height();
        // The halo reaches past the copy, within the picture.
        let halo = if lens.glass { (params.lens_rim * 0.28).ceil() as i32 } else { 0 };
        let [x, y, w, h] = placement.on_target.map(|v| v as i32);
        let [px, py, pw, ph] = placement.picture.map(|v| v as i32);
        let (x0, y0) = ((x - halo).max(px).max(0), (y - halo).max(py).max(0));
        let (x1, y1) = ((x + w + halo).min(px + pw), (y + h + halo).min(py + ph));
        if x1 <= x0 || y1 <= y0 { return; }
        let offset = placement.index as u64 * PARAMS_STRIDE;
        cx.queue.write_buffer(&self.copy_params[cx.slot], offset, bytemuck::bytes_of(&params));
        let source = source.create_view(&Default::default());
        let group = cx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("screen copy lens"), layout: &self.copy_layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&source) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.copy_params[cx.slot], offset: 0, size: NonZeroU64::new(size_of::<CopyLensParams>() as u64) }) },
        ] });
        pass.set_pipeline(&self.pipeline(cx.device, placement.format, true));
        pass.set_bind_group(0, &group, &[offset as u32]);
        pass.set_scissor_rect(x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32);
        pass.draw(0..3, 0..1);
    }
}

/// Four narrow straight edges and four corner squares, with no overlap.
fn edge_rects(c: &Camera, guard: i32) -> [[i32; 4]; 8] {
    let l = Lens::of(c.effect.unwrap());
    let (x, y, w, h) = (c.origin.0, c.origin.1, c.viewport.0 as i32, c.viewport.1 as i32);
    let halo = if l.glass { (l.rim * 0.28).ceil() as i32 } else { 0 };
    let a = (l.radius.max(l.rim).ceil() as i32 + 1 + guard).min(w.min(h) / 2);
    let e = (l.rim.ceil() as i32 + 1 + guard).min(a);
    [[x+a,y-halo,w-2*a,e+halo], [x+a,y+h-e,w-2*a,e+halo],
     [x-halo,y+a,e+halo,h-2*a], [x+w-e,y+a,e+halo,h-2*a],
     [x-halo,y-halo,a+halo,a+halo], [x+w-a,y-halo,a+halo,a+halo],
     [x-halo,y+h-a,a+halo,a+halo], [x+w-a,y+h-a,a+halo,a+halo]]
}
