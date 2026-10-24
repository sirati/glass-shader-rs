//! Magnifier lenses and glass dividers drawn into every window format, at
//! pixel scales 1 and 2: their rim pipelines must be made for the format
//! of the pass they are drawn in (the internal canvas, or the window when
//! a scaled picture is presented), not for one fixed format.
use glass::Lens;
use tile_render::{AtlasBuilder, Camera, ColorModel, Grid, Headless, LayerSpec, Layout, Present, RgbaImage, SpriteSets, TileRenderer, TileWorld};

/// Red left half, blue right half: a red/blue swap shows at once.
fn fixture(color: ColorModel) -> (tile_render::Atlas, TileWorld) {
    let mut atlas = AtlasBuilder::new(256);
    let mut sets = SpriteSets::default();
    let tile = RgbaImage::from_fn(64, 64, |x, y| if x < 32 { [200, 40, 30, 255] } else if y < 32 { [30, 50, 210, 255] } else { [40, 160, 60, 255] });
    let tile = sets.single(atlas.add(&tile, (0, 0)).unwrap());
    let grid = Grid { cols: 16, rows: 16, layout: Layout::Orthogonal, step: (64, 64), wrap_x: true, wrap_y: true };
    let mut world = TileWorld::new(grid, sets);
    world.color = color;
    let layer = world.add_layer(LayerSpec::new("halves", vec![tile]));
    for y in 0..grid.rows { for x in 0..grid.cols { world.set(layer, x, y, 0); } }
    (atlas.finish(), world)
}

#[derive(Clone, Copy, Debug)]
enum Placement { Mouse, Centre, Corner, Strip }

/// The window's cameras at pixel scale `scale` (canvas pixels): the
/// world, a zoom lens where `placement` puts it, and optionally the glass
/// dividers of a split.
fn cameras(canvas: (u32, u32), scale: u32, placement: Placement, rounded: bool, glass: bool, dividers: bool) -> Vec<Camera> {
    let (w, h) = (canvas.0 as i32, canvas.1 as i32);
    let mut cameras = vec![Camera::new(canvas)];
    let s = scale as f32;
    let lens = Lens { radius: if rounded { 24.0 / s } else { 0.0 }, rim: if glass { 14.0 } else { 2.0 } / s, glass };
    let (lw, lh) = (w * 2 / 5, h * 2 / 5);
    let origin = match placement {
        // About the pointer, partly off the right edge.
        Placement::Mouse => (w - lw / 2, h / 3),
        Placement::Centre => ((w - lw) / 2, (h - lh) / 2),
        Placement::Corner => (-lw / 4, -lh / 4),
        Placement::Strip => (0, h - lh / 2),
    };
    let (vw, vh) = if matches!(placement, Placement::Strip) { (w as u32, (lh / 2) as u32) } else { (lw as u32, lh as u32) };
    cameras.push(Camera { x: 200, y: 150, zoom: (2, 1), origin, effect: Some(lens.into()), ..Camera::new((vw, vh)) });
    if dividers {
        let t = (16 / scale) as i32;
        let bar = |[x, y, bw, bh]: [i32; 4]| Camera { origin: (x, y), cells: false,
            effect: Some(Lens { radius: t as f32 / 2.0, rim: t as f32 / 2.0, glass: true }.into()),
            target_clip: Some([x, y.max(0), x + bw, y + bh]), ..Camera::new((bw as u32, bh as u32)) };
        cameras.push(bar([w / 2 - t / 2, 0, t, h]));
        cameras.push(bar([-t, h / 2, w + 2 * t, t]));
    }
    cameras
}

const FORMATS: [wgpu::TextureFormat; 4] = [wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Bgra8Unorm,
    wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureFormat::Bgra8UnormSrgb];

fn target(gpu: &Headless, format: wgpu::TextureFormat, size: (u32, u32), copyable: bool) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("window"), size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format,
        // (A surface may not be copyable: then it is not read back.)
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | if copyable { wgpu::TextureUsages::COPY_SRC } else { wgpu::TextureUsages::empty() },
        view_formats: &[],
    })
}

/// Draws one frame (and a replay of the cached picture), returning the
/// validation error if any and, from a copyable target, the picture as RGBA.
fn draw(gpu: &Headless, renderer: &mut TileRenderer, target: &wgpu::Texture, cameras: &[Camera], present: Present, frame: u32) -> (Option<String>, Option<RgbaImage>) {
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    renderer.render_presented(&gpu.device, &gpu.queue, &mut encoder, target, cameras, &[], frame, [0, 0, 0, 255], present);
    renderer.render_cached_picture(&gpu.device, &gpu.queue, &mut encoder, target, &[], [0, 0, 0, 255], present);
    let image = if target.usage().contains(wgpu::TextureUsages::COPY_SRC) {
        Some(tile_render::gpu::read_texture(&gpu.device, &gpu.queue, encoder, target))
    } else {
        gpu.queue.submit([encoder.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        None
    };
    renderer.controls_submitted();
    let error = pollster::block_on(scope.pop()).map(|e| e.to_string());
    (error, image)
}

fn decoded(format: wgpu::TextureFormat, [r, g, b, a]: [u8; 4]) -> [u8; 4] {
    // `read_texture` swaps only `Bgra8Unorm`; do it for the sRGB one too.
    if format == wgpu::TextureFormat::Bgra8UnormSrgb { [b, g, r, a] } else { [r, g, b, a] }
}

/// Window pixels clear of every lens placement and divider (640x400 at
/// scale 1 or 2) showing the red and the blue quarter of a tile.
const RED: (u32, u32) = (8, 264);
const BLUE: (u32, u32) = (100, 264);

#[test]
fn lens_and_glass_draw_into_every_window_format_at_every_pixel_scale() {
    let gpu = Headless::new().expect("Vulkan GPU");
    let window = (640, 400);
    let mut failures = Vec::new();
    let mut runs = 0;
    for color in [ColorModel::Direct, ColorModel::Quantized { bits: 5 }] {
        let (atlas, world) = fixture(color);
        for format in FORMATS {
            let mut renderer = TileRenderer::new(&gpu.device, format, SHADERS);
            renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
            renderer.set_world(&gpu.device, &world);
            let mut frame = 0;
            for scale in [1, 2] {
                let canvas = (window.0 / scale, window.1 / scale);
                let present = Present { canvas, zoom: (scale, 1), at: (0, 0), target: window };
                for copyable in [false, true] {
                    let target = target(&gpu, format, window, copyable);
                    for placement in [Placement::Mouse, Placement::Centre, Placement::Corner, Placement::Strip] {
                        for rounded in [false, true] { for glass in [false, true] { for dividers in [false, true] {
                            let cameras = cameras(canvas, scale, placement, rounded, glass, dividers);
                            frame += 1;
                            runs += 1;
                            let (error, image) = draw(&gpu, &mut renderer, &target, &cameras, present, frame);
                            let what = format!("{color:?} {format:?} scale {scale} copyable {copyable} {placement:?} rounded {rounded} glass {glass} dividers {dividers}");
                            if let Some(e) = error { failures.push(format!("{what}: {e}")); continue; }
                            // Away from the lens and the dividers: the world's red and blue.
                            let Some(image) = image else { continue };
                            let (red, blue) = (decoded(format, image.get(RED.0, RED.1)), decoded(format, image.get(BLUE.0, BLUE.1)));
                            // (sRGB targets brighten every channel; neither swaps them.)
                            if red[0] < red[2] + 100 { failures.push(format!("{what}: red reads {red:?}")); }
                            if blue[2] < blue[0] + 100 { failures.push(format!("{what}: blue reads {blue:?}")); }
                        } } }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} of {runs} frames failed:\n{}", failures.len(), failures.join("\n"));
}

/// A renderer made for one format drawing into a window of the other (a
/// surface reconfigured), and back.
#[test]
fn a_renderer_follows_its_target_to_another_format() {
    let gpu = Headless::new().expect("Vulkan GPU");
    let window = (640, 400);
    let (atlas, world) = fixture(ColorModel::Quantized { bits: 5 });
    let mut renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
    renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
    renderer.set_world(&gpu.device, &world);
    let mut frame = 0;
    for format in [wgpu::TextureFormat::Bgra8Unorm, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Bgra8Unorm] {
        for scale in [1, 2] {
            let canvas = (window.0 / scale, window.1 / scale);
            let present = Present { canvas, zoom: (scale, 1), at: (0, 0), target: window };
            frame += 1;
            let (error, image) = draw(&gpu, &mut renderer, &target(&gpu, format, window, true), &cameras(canvas, scale, Placement::Mouse, true, true, true), present, frame);
            assert_eq!(error, None, "{format:?} scale {scale}");
            let red = image.unwrap().get(RED.0, RED.1);
            assert!(red[0] > 150 && red[2] < 80, "{format:?} scale {scale}: red {red:?}");
        }
    }
}

/// Review pictures: the zoom at the mouse at pixel scale 2, in both
/// window formats (`LENS_FORMAT_SHOTS=dir`).
#[test]
fn zoom_at_mouse_at_pixel_scale_2_looks_the_same_in_both_formats() {
    let gpu = Headless::new().expect("Vulkan GPU");
    let window = (640, 400);
    let (atlas, world) = fixture(ColorModel::Quantized { bits: 5 });
    let mut pictures = Vec::new();
    for format in [wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureFormat::Bgra8Unorm] {
        let mut renderer = TileRenderer::new(&gpu.device, format, SHADERS);
        renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
        renderer.set_world(&gpu.device, &world);
        let canvas = (window.0 / 2, window.1 / 2);
        let present = Present { canvas, zoom: (2, 1), at: (0, 0), target: window };
        let cameras = cameras(canvas, 2, Placement::Mouse, true, true, true);
        let (error, image) = draw(&gpu, &mut renderer, &target(&gpu, format, window, true), &cameras, present, 1);
        assert_eq!(error, None, "{format:?}");
        let image = image.unwrap();
        if let Ok(dir) = std::env::var("LENS_FORMAT_SHOTS") {
            std::fs::create_dir_all(&dir).unwrap();
            image::save_buffer(std::path::Path::new(&dir).join(format!("zoom-at-mouse-scale2-{format:?}.png")), bytemuck::cast_slice(&image.pixels), image.width, image.height, image::ColorType::Rgba8).unwrap();
        }
        pictures.push(image);
    }
    let differences = pictures[0].pixels.iter().zip(&pictures[1].pixels).filter(|(a, b)| a != b).count();
    assert_eq!(differences, 0, "Rgba8Unorm and Bgra8Unorm pictures differ in {differences} pixels");
    // The red and blue halves read as red and blue.
    let red = pictures[0].get(RED.0, RED.1);
    let blue = pictures[0].get(BLUE.0, BLUE.1);
    assert!(red[0] > 150 && red[2] < 80, "red {red:?}");
    assert!(blue[2] > 150 && blue[0] < 80, "blue {blue:?}");
}

/// The viewer's shader module (it holds the lens's shaders), drawing with the lens.
const SHADERS: tile_render::Shaders = tile_render::Shaders::new(viewer_spirv::SPIRV, viewer_spirv::SPIRV_FALLBACK).with_effect(glass::effect);
