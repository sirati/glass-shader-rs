//! Rounded magnifiers, bounded glass refraction and review images.
use glass::Lens;
use tile_render::{AtlasBuilder, Camera, ColorModel, Grid, Headless, LayerSpec, Layout, RgbaImage, SpriteSets, TileRenderer, TileWorld};

fn fixture(color: ColorModel) -> (tile_render::Atlas, TileWorld) {
    let mut atlas = AtlasBuilder::new(256);
    let mut sets = SpriteSets::default();
    let tiles: Vec<_> = (0..4).map(|kind| {
        let tile = RgbaImage::from_fn(64, 32, |x, y| {
            let dx = (x as i32 - 32).abs();
            let dy = (y as i32 - 16).abs();
            if dx + 2*dy > 32 { return [0; 4]; }
            let grain = ((x*13+y*37) % 13) as u8;
            let base: [u8; 3] = match kind { 0 => [47, 93, 128], 1 => [78, 109, 47], 2 => [133, 130, 67], _ => [63, 97, 43] };
            if kind != 0 && (y as i32 - x as i32 / 2).abs() < 2 { [177, 157, 104, 255] }
            else { [base[0]+grain, base[1]+grain, base[2]+grain, 255] }
        });
        sets.single(atlas.add(&tile, (0, 0)).unwrap())
    }).collect();
    let grid = Grid { cols: 32, rows: 48, layout: Layout::Staggered, step: (32, 16), wrap_x: true, wrap_y: true };
    let mut world = TileWorld::new(grid, sets);
    world.color = color;
    let layer = world.add_layer(LayerSpec::new("terrain", tiles));
    for y in 0..grid.rows { for x in 0..grid.cols { world.set(layer, x, y, if x % 9 < 2 { 0 } else { 1+(x*3+y/3)%3 }); } }
    (atlas.finish(), world)
}

/// Straight survey lines make bevel compression and corner bending visible.
fn survey_grid() -> (tile_render::Atlas, TileWorld) {
    survey_pattern(true)
}

fn survey_pattern(lines: bool) -> (tile_render::Atlas, TileWorld) {
    let mut builder = AtlasBuilder::new(256);
    let tile = RgbaImage::from_fn(128,128, |x,y| {
        if lines && (2*y as i32 - x as i32 - 32).rem_euclid(128) <= 1 { [215,225,215,255] }
        else { [38,67,49,255] }
    });
    let mut sets = SpriteSets::default();
    let tile = sets.single(builder.add(&tile,(0,0)).unwrap());
    let grid = Grid { cols: 32, rows: 24, layout: Layout::Orthogonal, step: (128,128), wrap_x: true, wrap_y: true };
    let mut world = TileWorld::new(grid, sets);
    let layer = world.add_layer(LayerSpec::new("survey grid", vec![tile]));
    for y in 0..grid.rows { for x in 0..grid.cols { world.set(layer,x,y,0); } }
    (builder.finish(),world)
}

fn cameras(scale: u32, rounded: bool, glass: bool, background: bool) -> Vec<Camera> {
    let mut cameras = Vec::new();
    if background { cameras.push(Camera::new((1024/scale, 640/scale))); }
    cameras.push(Camera { x: 170, y: 180, zoom: (2, 1), origin: (170/scale as i32, 150/scale as i32),
        effect: Some(Lens { radius: if rounded { 25.0/scale as f32 } else { 0.0 }, rim: if glass { 14.0 } else { 2.0 } / scale as f32, glass }.into()),
        ..Camera::new((684/scale, 330/scale)) });
    cameras
}

#[test]
fn rounded_corners_preserve_the_backdrop_and_glass_lights_black() {
    let gpu = Headless::new().expect("Vulkan GPU");
    eprintln!("lens adapter: {:?}", gpu.adapter.get_info());
    for color in [ColorModel::Direct, ColorModel::Quantized { bits: 5 }] {
        let (atlas, world) = fixture(color);
        let mut renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
        renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
        renderer.set_world(&gpu.device, &world);
        for scale in [1, 2] { for rounded in [false, true] { for glass in [false, true] { for background in [false, true] {
            let cameras = cameras(scale, rounded, glass, background);
            let size = (1024/scale, 640/scale);
            let expected = REFERENCE.render_views(&world, &atlas, &cameras, size, 0, [0, 0, 0, 255]);
            let actual = gpu.render_views(&mut renderer, &cameras, size, 0, [0, 0, 0, 255]);
            let mut differences = 0;
            let mut worst = 0;
            for (g, c) in actual.pixels.iter().zip(&expected.pixels) { for channel in 0..3 {
                worst = worst.max(g[channel].abs_diff(c[channel]));
                if g[channel].abs_diff(c[channel]) > if color == ColorModel::Direct { 1 } else { 9 } { differences += 1; }
            } }
            assert_eq!(differences, 0, "{color:?} scale {scale} rounded {rounded} glass {glass} background {background}, worst {worst}");
            let lens = cameras.last().unwrap();
            let (x, y) = (lens.origin.0 as u32, lens.origin.1 as u32);
            if rounded && !background { assert_eq!(actual.get(x, y), [0, 0, 0, 255]); }
            if glass && !background { assert!(actual.get(x + lens.viewport.0/2, y)[0] > 0); }
            let mut square = cameras.clone(); square.last_mut().unwrap().effect = None;
            let unframed = REFERENCE.render_views(&world, &atlas, &square, size, 0, [0, 0, 0, 255]);
            assert_eq!(actual.get(x + lens.viewport.0/2, y + lens.viewport.1/2), unframed.get(x + lens.viewport.0/2, y + lens.viewport.1/2));
            if color == ColorModel::Direct && rounded && glass {
                if let Some(dir) = std::env::var_os("C3R_ZOOM_SHOTS") {
                    std::fs::create_dir_all(&dir).unwrap();
                    let path = std::path::Path::new(&dir).join(format!("lens-{}-scale{scale}.png", if background { "map" } else { "black" }));
                    let image = if scale == 1 { actual.clone() } else { RgbaImage::from_fn(1024, 640, |x, y| actual.get(x/scale, y/scale)) };
                    image::save_buffer(path, bytemuck::cast_slice(&image.pixels), image.width, image.height, image::ColorType::Rgba8).unwrap();
                }
            }
        } } } }
    }
}

#[test]
fn clipped_moving_lenses_keep_their_shape_and_casting_marks() {
    let gpu = Headless::new().expect("Vulkan GPU");
    let (atlas, world) = fixture(ColorModel::Direct);
    let mut renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
    renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
    renderer.set_world(&gpu.device, &world);
    let lens = Camera { cells: false, origin: (20, 30), effect: Some(Lens { radius: 24.0, rim: 14.0, glass: true }.into()), ..Camera::new((120, 90)) };
    let first = gpu.render_views(&mut renderer, &[lens], (240, 180), 0, [0, 0, 0, 255]);
    let moved = Camera { origin: (70, 60), ..lens };
    let second = gpu.render_views(&mut renderer, &[moved], (240, 180), 0, [0, 0, 0, 255]);
    for y in 0..90 { for x in 0..120 { assert_eq!(first.get(x+20, y+30), second.get(x+70, y+60)); } }
    for origin in [(-25, -20), (180, 140)] {
        let clipped = Camera { origin, target_clip: Some([5, 8, 235, 175]), ..lens };
        let expected = REFERENCE.render_views(&world, &atlas, &[clipped], (240, 180), 0, [0, 0, 0, 255]);
        let actual = gpu.render_views(&mut renderer, &[clipped], (240, 180), 0, [0, 0, 0, 255]);
        for (a, b) in actual.pixels.iter().zip(expected.pixels) { for c in 0..3 { assert!(a[c].abs_diff(b[c]) <= 1); } }
    }
}

/// Reproducible opt-in measurement. Frame timestamps include the backdrop
/// copy; the separate rim timestamp measures the eight bounded edge draws.
#[test]
#[ignore = "GPU cost measurement; run without concurrent GPU tests"]
fn measure_lens_gpu_cost_at_4k() {
    let gpu = Headless::new().expect("Vulkan GPU");
    eprintln!("lens benchmark adapter: {:?}", gpu.adapter.get_info());
    let (atlas, world) = fixture(ColorModel::Quantized { bits: 5 });
    let mut renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
    renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
    renderer.set_world(&gpu.device, &world);
    assert!(renderer.enable_timing(&gpu.device, &gpu.queue));
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("4K lens timing"), size: wgpu::Extent3d { width: 3840, height: 2160, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
    });
    for scale in [1, 2] {
        let (mut spans, mut rims): ([Vec<f64>; 2], [Vec<f64>; 2]) = Default::default();
        let mut increments = Vec::new();
        // Pair baseline/glass frames and alternate their order. Separate long
        // blocks can mistake clock or desktop-load drift for extra glass cost.
        for frame in 0..48 {
            let mut paired = [None; 2];
            for glass in if frame % 2 == 0 { [false, true] } else { [true, false] } {
                let c = Camera { origin: (768/scale, 216/scale), zoom: (2, 1), x: 170, y: 180,
                    effect: glass.then_some(Lens { radius: 56.0/scale as f32, rim: 24.0/scale as f32, glass: true }.into()),
                    ..Camera::new((2304/scale as u32, 432/scale as u32)) };
                let cams = [Camera::new((3840/scale as u32, 2160/scale as u32)), c];
                let present = tile_render::Present { canvas: (3840/scale as u32, 2160/scale as u32), target: (3840, 2160), zoom: (scale as u32, 1), at: (0, 0) };
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                renderer.render_presented(&gpu.device, &gpu.queue, &mut encoder, &target, &cams, &[], frame, [0,0,0,255], present);
                renderer.timer().unwrap().finish(&mut encoder);
                gpu.queue.submit([encoder.finish()]);
                renderer.controls_submitted();
                renderer.timer().unwrap().submitted();
                gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                for times in renderer.timer().unwrap().collect(&gpu.device) {
                    if frame >= 8 {
                        let index = usize::from(glass);
                        spans[index].push(times.span);
                        rims[index].push(times.passes.iter().filter(|(label, _)| label.ends_with("lens rim")).map(|(_, ms)| *ms).sum::<f64>());
                        paired[index] = Some(times.span);
                    }
                }
            }
            if let [Some(base), Some(glass)] = paired { increments.push(glass - base); }
        }
        for glass in [false, true] {
            let index = usize::from(glass);
            spans[index].sort_by(f64::total_cmp); rims[index].sort_by(f64::total_cmp);
            eprintln!("4K scale {scale}, glass {glass}: frame median {:.4} ms, rim median {:.4} ms, {} samples", spans[index][spans[index].len()/2], rims[index][rims[index].len()/2], spans[index].len());
        }
        increments.sort_by(f64::total_cmp);
        eprintln!("4K scale {scale}: paired increment median {:.4} ms, p10/p90 {:.4}/{:.4} ms, {} pairs",
            increments[increments.len()/2], increments[increments.len()/10], increments[increments.len()*9/10], increments.len());
    }
}

/// Track a whole line across the bevel at both presentation scales. The
/// face and rim must meet continuously, then gain tens of pixels of bend.
#[test]
fn straight_lines_bend_continuously_across_the_bevel_at_both_scales() {
    let gpu = Headless::new().expect("Vulkan GPU");
    let (atlas, world) = survey_grid();
    let mut renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
    renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
    renderer.set_world(&gpu.device, &world);
    let (field_atlas, field_world) = survey_pattern(false);
    let mut field_renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
    field_renderer.set_atlas(&gpu.device, &gpu.queue, &field_atlas);
    field_renderer.set_world(&gpu.device, &field_world);
    for scale in [1, 2] { for menu in [false, true] {
        let lens = Camera { cells: false, origin: (40/scale,40/scale),
            effect: Some(Lens { radius: 56.0/scale as f32, rim: 24.0/scale as f32, glass: true }.into()),
            ..Camera::new((600/scale as u32,200/scale as u32)) };
        let present = tile_render::Present { canvas: (700/scale as u32,300/scale as u32), target: (700,300), zoom: (scale as u32,1), at: (0,0) };
        let base = Camera::new(present.canvas);
        let (cameras, copies) = if menu {
            let copy = tile_render::ScreenCopy { source: [100/scale,40/scale,300/scale,100/scale], at: lens.origin, zoom: (2,1), effect: lens.effect };
            (vec![base], vec![copy])
        } else { (vec![base,lens], Vec::new()) };
        let image = gpu.render_presented(&mut renderer, &cameras, &copies, present, 0, [0,0,0,255]);
        let plain_cameras: Vec<_> = cameras.iter().map(|c| Camera { effect: None, ..*c }).collect();
        let plain_copies: Vec<_> = copies.iter().map(|c| tile_render::ScreenCopy { effect: None, ..*c }).collect();
        let plain = gpu.render_presented(&mut renderer, &plain_cameras, &plain_copies, present, 0, [0,0,0,255]);
        let field = gpu.render_presented(&mut field_renderer, &cameras, &copies, present, 0, [0,0,0,255]);
        // Subtract the same green field without lines, including its coloured
        // glass lighting. A black control assumes a channel-wise linear blend
        // and mistakes colour-preserving radiance compression for dispersion.
        let contrast = |x,y,ch: usize| {
            let base = field.get(x,y);
            let headroom = 255.0 - *base[..3].iter().max().unwrap() as f32;
            (image.get(x,y)[ch] as f32 - base[ch] as f32) / headroom.max(1.0)
        };
        let mut max_fringe_chroma = 0.0_f32;
        for y in 40..96 { for x in 40..96 {
            let pixel = image.get(x,y);
            assert!(*pixel[..3].iter().min().unwrap() < 250,
                "scale {scale}, menu {menu}: a refracted line clips the corner highlight at {x}/{y}");
            // Measure displayed chroma directly. Inverting a channel-wise
            // white blend is invalid for a shared, colour-preserving shoulder.
            let channels = [pixel[0] as f32/255.0, pixel[1] as f32/255.0, pixel[2] as f32/255.0];
            let lo = channels.iter().copied().fold(1.0_f32, f32::min);
            let hi = channels.iter().copied().fold(0.0_f32, f32::max);
            max_fringe_chroma = max_fringe_chroma.max(hi-lo);
        } }
        assert!(max_fringe_chroma < 0.22,
            "scale {scale}, menu {menu}: saturated corner dashes, chroma {max_fringe_chroma}");
        let intercept = if menu { -80 + 192*scale } else { 96*scale };
        let displacement = |y| {
            let baseline = (2*y as i32 + intercept) as u32;
            let peak = (baseline-4..baseline+124).max_by(|&a,&b| contrast(a,y,1).total_cmp(&contrast(b,y,1))).unwrap();
            // Boxed menu art can give a line several equal peak pixels. Its
            // contrast centroid avoids jumping between opposite plateau ends.
            let floor = (baseline-4..baseline+124).map(|x| contrast(x,y,1)).min_by(f32::total_cmp).unwrap();
            let (mut weight, mut moment) = (0.0, 0.0);
            for x in peak-8..=peak+8 {
                let value = (contrast(x,y,1) - floor).max(0.0);
                weight += value;
                moment += value * (x as i32-baseline as i32) as f32;
            }
            let bent = moment / weight;
            // Measure against the same boxed source phase. The analytic line
            // intercept can be several pixels from the centre of a thick box.
            let floor = (baseline-12..=baseline+12).map(|x| plain.get(x,y)[1]).min().unwrap();
            let (mut weight, mut moment) = (0.0, 0.0);
            for x in baseline-12..=baseline+12 {
                let value = plain.get(x,y)[1].saturating_sub(floor) as f32;
                weight += value;
                moment += value * (x as i32-baseline as i32) as f32;
            }
            bent - moment / weight
        };
        let shifts: Vec<_> = [63,60,57,54,51,48,45,42,41].into_iter().map(displacement).collect();
        eprintln!("scale {scale}, menu {menu}, line displacement across 24px bevel: {shifts:?}");
        assert!(shifts[0].abs() <= 3.0, "scale {scale}, menu {menu}, clear face continuity: {shifts:?}");
        assert!(shifts[4] >= 12.0 && shifts[8] >= 85.0, "scale {scale}, menu {menu}, broad visible refraction: {shifts:?}");
        // Allow half a physical pixel for the centroid of boxed source art.
        // The shared analytic corner-ray test checks exact monotonicity.
        for pair in shifts.windows(2) { assert!(pair[1] + 0.5 >= pair[0], "scale {scale}, menu {menu}, continuously increasing bend: {shifts:?}"); }
        let baseline = (82 + intercept) as u32;
        // Integrated edge positions resolve subtle dispersion even when both
        // channel peaks occupy the same pixel. Isolated RGB rays used to move
        // these peaks several pixels apart and produce coloured dashes.
        let centroid = |c| {
            let floor = (baseline+70..baseline+124).map(|x| contrast(x,41,c)).min_by(f32::total_cmp).unwrap();
            let (mut weight, mut moment) = (0.0, 0.0);
            for x in baseline+70..baseline+124 {
                let value = (contrast(x,41,c) - floor).max(0.0);
                weight += value;
                moment += value * (x-baseline) as f32;
            }
            moment / weight
        };
        let fringe = (centroid(0) - centroid(2)).abs();
        eprintln!("scale {scale}, menu {menu}, red/blue centroid separation: {fringe:.3}px");
        assert!((0.03..=1.5).contains(&fringe), "scale {scale}, menu {menu}, subtle red/blue fringe: {fringe}px");
    } }
}

/// Empty black glass isolates lighting and physical silhouette coverage from
/// the pixelated art. Scale 2 must not turn the corner into 2x2 stair steps.
#[test]
fn presented_glass_keeps_physical_width_and_smooth_corners() {
    let gpu = Headless::new().expect("Vulkan GPU");
    let (atlas, world) = fixture(ColorModel::Direct);
    let mut renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
    renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
    renderer.set_world(&gpu.device, &world);
    let render = |renderer: &mut TileRenderer, scale: u32| {
        let c = Camera { cells: false, origin: (40/scale as i32,40/scale as i32),
            effect: Some(Lens { radius: 56.0/scale as f32, rim: 24.0/scale as f32, glass: true }.into()),
            ..Camera::new((600/scale,200/scale)) };
        gpu.render_presented(renderer, &[c], &[], tile_render::Present {
            canvas: (700/scale,300/scale), target: (700,300), zoom: (scale,1), at: (0,0) }, 0, [0,0,0,255])
    };
    let full = render(&mut renderer, 1);
    let boxed = render(&mut renderer, 2);
    for (i,(a,b)) in full.pixels.iter().zip(&boxed.pixels).enumerate() { for c in 0..3 {
        assert!(a[c].abs_diff(b[c]) <= 1, "window-pixel optics at {}/{}: {a:?} vs {b:?}", i%700, i/700);
    } }
    assert!((40..64).map(|y| boxed.get(300,y)[2]).max().unwrap() >= 100, "visible unseen-light streak on black");
    assert_eq!(boxed.get(300,64), [0,0,0,255], "the face stays clear");
    assert_ne!(boxed.get(56,56), boxed.get(57,56), "the corner is shaded per physical pixel");
    for image in [&full, &boxed] {
        for profile in [
            (216..240).map(|y| image.get(340,y)).collect::<Vec<_>>(),
            (616..640).map(|x| image.get(x,140)).collect(),
        ] {
            assert!(profile.iter().any(|p| p[0].max(p[2]) >= 100),
                "a secondary crest glint lights the bottom/right: {profile:?}");
            assert!(profile.iter().filter(|p| p[0].max(p[2]) >= 40).count() >= 4,
                "bottom/right light spans several physical pixels: {profile:?}");
        }
    }
}

/// A floating world lens sits below driver UI. Final-resolution shading
/// must preserve both opaque/translucent later pixels and cached replays.
#[test]
fn window_glass_preserves_later_interface_views_and_cached_frames() {
    let gpu = Headless::new().expect("Vulkan GPU");
    for color in [ColorModel::Direct, ColorModel::Quantized { bits: 5 }] {
        let (atlas, world) = fixture(color);
        let mut renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
        renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
        renderer.set_world(&gpu.device, &world);
        let lens = Camera { cells: false, origin: (20,20),
            effect: Some(Lens { radius: 28.0, rim: 12.0, glass: true }.into()), ..Camera::new((300,100)) };
        let ui = Camera { origin: (60,20), ..Camera::new((100,24)) };
        let before = gpu.render_views(&mut renderer, &[lens], (350,150), 0, [0,0,0,255]);
        let after = gpu.render_views(&mut renderer, &[lens,ui], (350,150), 0, [0,0,0,255]);
        let present = tile_render::Present { canvas: (350,150), target: (714,318), zoom: (2,1), at: (7,9) };
        let actual = gpu.render_presented(&mut renderer, &[lens,ui], &[], present, 0, [0,0,0,255]);
        let cached = gpu.render_presented(&mut renderer, &[lens,ui], &[], present, 0, [0,0,0,255]);
        assert_eq!(actual, cached, "cached lens uniforms and snapshots");
        let mut covered = 0;
        for y in 20..32 { for x in 60..160 {
            if before.get(x,y) == after.get(x,y) { continue; }
            covered += 1;
            for dy in 0..2 { for dx in 0..2 {
                let pixel = actual.get(7+x*2+dx,9+y*2+dy);
                for c in 0..3 { assert!(pixel[c].abs_diff(after.get(x,y)[c]) <= 1, "{color:?}: later UI at {x}/{y}: {pixel:?} vs {:?}", after.get(x,y)); }
            } }
        } }
        assert!(covered > 200, "UI must actually cover the top bevel");
    }
}

/// Check the black-background requirement independently of the CPU oracle:
/// luminous angular reflections, a clear face and no repeated edge marks.
#[test]
fn black_glass_has_a_curved_luminous_rim_without_periodic_ticks() {
    let gpu = Headless::new().expect("Vulkan GPU");
    let (atlas, world) = fixture(ColorModel::Direct);
    let mut renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
    renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
    renderer.set_world(&gpu.device, &world);
    let c = Camera { cells: false, origin: (40, 40), effect: Some(Lens { radius: 30.0, rim: 14.0, glass: true }.into()), ..Camera::new((600, 200)) };
    let image = gpu.render_views(&mut renderer, &[c], (700, 300), 0, [0,0,0,255]);
    assert_eq!(image.get(340, 140), [0,0,0,255], "clear, untinted face");
    assert!(image.get(340, 39)[2] > 0, "scattered light outside the silhouette");
    assert_eq!(image.get(340, 35), [0,0,0,255], "halo is bounded");
    let top: Vec<_> = (40..54).map(|y| image.get(300,y)[2]).collect();
    assert!(top.iter().max().unwrap() - top.iter().min().unwrap() > 15, "curved light profile: {top:?}");
    assert!(image.get(300,40)[2] > image.get(40,140)[2], "unseen light varies with rim angle");
    // Finite reflected lights vary smoothly along a straight edge. Only the
    // scratch and bubble may create sharp local changes on top; no ticks below.
    let mut runs = 0;
    let mut flaw = false;
    for x in 80..600 {
        let differs = (40..54).any(|y| (0..3).any(|ch|
            image.get(x,y)[ch].abs_diff(image.get(x-1,y)[ch]) > 8));
        if differs {
            assert!((211..219).contains(&x) || (444..457).contains(&x),
                "an unexpected sharp mark on the top rim at {x}");
        }
        if differs && !flaw { runs += 1; }
        flaw = differs;
    }
    assert!((2..=4).contains(&runs), "only the local scratch and bubble have sharp edges, got {runs} runs");
    for x in 80..600 { for y in 226..240 { for ch in 0..3 {
        assert!(image.get(x,y)[ch].abs_diff(image.get(x-1,y)[ch]) <= 2,
            "a plain black bottom edge has no periodic ticks at {x}/{y}");
    } } }
    // The bottom and right need an actual secondary reflection on pure black,
    // with multiple light bands and some colour, rather than a dim grey tube.
    for profile in [
        (226..241).map(|y| image.get(350,y)).collect::<Vec<_>>(),
        (626..641).map(|x| image.get(x,140)).collect(),
    ] {
        let peaks = profile.windows(3).filter(|p| p[1][2] > p[0][2] && p[1][2] > p[2][2]).count();
        assert!(peaks >= 2, "separate reflected bands on bottom/right: {profile:?}");
        assert!(profile.iter().any(|p| p[0].max(p[2]) >= 65), "visible secondary light: {profile:?}");
        assert!(profile.iter().any(|p| p[0].abs_diff(p[2]) >= 5), "tinted environment: {profile:?}");
    }
}

#[test]
fn screen_copy_lenses_preserve_corners_and_clear_menu_content() {
    use tile_render::{Present, ScreenCopy};
    let gpu = Headless::new().expect("Vulkan GPU");
    for color in [ColorModel::Direct, ColorModel::Quantized { bits: 5 }] {
        let (atlas, world) = fixture(color);
        let mut renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
        renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
        renderer.set_world(&gpu.device, &world);
        for scale in [1, 2] { for glass in [false, true] {
            let size = (800/scale, 600/scale);
            let cameras = [Camera::new(size)];
            let copy = ScreenCopy { source: [40/scale as i32, 100/scale as i32, 160/scale as i32, 120/scale as i32],
                at: (350/scale as i32, 200/scale as i32), zoom: (2,1),
                effect: Some(Lens { radius: 56.0/scale as f32, rim: 24.0/scale as f32, glass }.into()) };
            let actual = gpu.render_composed(&mut renderer, &cameras, &[copy], size, 0, [0,0,0,255]);
            let expected = REFERENCE.render_composed(&world, &atlas, &cameras, &[copy], size, 0, [0,0,0,255]);
            for (a,b) in actual.pixels.iter().zip(&expected.pixels) { for ch in 0..3 {
                assert!(a[ch].abs_diff(b[ch]) <= 1, "menu lens {color:?}, scale {scale}, glass {glass}: {a:?} vs {b:?}");
            } }
            let plain = gpu.render_composed(&mut renderer, &cameras, &[ScreenCopy { effect: None, ..copy }], size, 0, [0,0,0,255]);
            assert_eq!(actual.get(500/scale,300/scale), plain.get(500/scale,300/scale));
            let base = gpu.render_views(&mut renderer, &cameras, size, 0, [0,0,0,255]);
            assert_eq!(actual.get(350/scale,200/scale), base.get(350/scale,200/scale));
            // The final-window pass keeps the physical lens width when pixels are boxed.
            let present = Present { canvas: size, target: (800,600), zoom: (scale,1), at: (0,0) };
            let presented = gpu.render_presented(&mut renderer, &cameras, &[copy], present, 0, [0,0,0,255]);
            assert_eq!(presented.get(500,300), plain.get(500/scale,300/scale));
            let boxed = gpu.render_presented(&mut renderer, &[cameras[0].boxed(scale)], &[copy.boxed(scale)], Present::unscaled((800,600)), 0, [0,0,0,255]);
            // Background art is sampled differently, but the silhouette is identical.
            assert_eq!(presented.get(350,200), boxed.get(350,200));
        } }
    }
}

#[test]
#[ignore = "4K review captures; set C3R_ZOOM_SHOTS"]
fn capture_glass_review_at_4k() {
    let dir = std::path::PathBuf::from(std::env::var_os("C3R_ZOOM_SHOTS").expect("C3R_ZOOM_SHOTS"));
    std::fs::create_dir_all(&dir).unwrap();
    // Move the repeating terrain's phase to distinguish source art from any
    // marks generated by the rim. Override for a reproducible comparison.
    let phase = std::env::var("C3R_ZOOM_PHASE").unwrap_or_else(|_| "181,187".into());
    let (phase_x, phase_y) = phase.split_once(',').expect("C3R_ZOOM_PHASE=x,y");
    let (phase_x, phase_y) = (phase_x.parse::<i32>().unwrap(), phase_y.parse::<i32>().unwrap());
    eprintln!("terrain capture phase: {phase_x},{phase_y}; survey phase: 200,180");
    let gpu = Headless::new().expect("Vulkan GPU");
    let mut renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
    let save = |name: String, image: &RgbaImage| image::save_buffer(dir.join(name), bytemuck::cast_slice(&image.pixels), image.width, image.height, image::ColorType::Rgba8).unwrap();
    for grid in [false, true] {
      let (atlas, world) = if grid { survey_grid() } else { fixture(ColorModel::Direct) };
      renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
      renderer.set_world(&gpu.device, &world);
      for scale in [1, 2] { for background in [false, true] {
        if grid && !background { continue; }
        let mut cams = Vec::new();
        if background { cams.push(Camera::new((3840/scale,2160/scale))); }
        cams.push(Camera { x: if grid { 200 } else { phase_x }, y: if grid { 180 } else { phase_y }, zoom: (2,1), origin: (768/scale as i32,216/scale as i32),
            effect: Some(Lens { radius: 56.0/scale as f32, rim: 24.0/scale as f32, glass: true }.into()), ..Camera::new((2304/scale,432/scale)) });
        let image = gpu.render_presented(&mut renderer, &cams, &[], tile_render::Present { canvas: (3840/scale,2160/scale), target: (3840,2160), zoom: (scale,1), at: (0,0) }, 0, [0,0,0,255]);
        let background = if grid { "grid" } else if background { "map" } else { "black" };
        save(format!("glass6-{background}-4k-scale{scale}.png"), &image);
        let crop = RgbaImage::from_fn(420,330, |x,y| image.get(752+x/3,200+y/3));
        save(format!("glass6-{background}-corner-3x-scale{scale}.png"), &crop);
        if !grid && background == "black" {
            let mut empty = *cams.last().unwrap(); empty.cells = false;
            // Independently build and render each empty-black scale. Identical
            // output is valid for window-pixel optics, but cannot be a cached
            // scale-1 image or an accidentally copied file.
            let mut empty_renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
            empty_renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
            empty_renderer.set_world(&gpu.device, &world);
            eprintln!("independent empty-black scale {scale}: canvas {}x{}, lens {}x{}, rim {} composition pixels",
                3840/scale, 2160/scale, empty.viewport.0, empty.viewport.1, Lens::of(empty.effect.unwrap()).rim);
            let image = gpu.render_presented(&mut empty_renderer, &[empty], &[], tile_render::Present { canvas: (3840/scale,2160/scale), target: (3840,2160), zoom: (scale,1), at: (0,0) }, scale, [0,0,0,255]);
            save(format!("glass6-empty-black-4k-scale{scale}.png"), &image);
            let crop = RgbaImage::from_fn(420,330, |x,y| image.get(752+x/3,200+y/3));
            save(format!("glass6-empty-black-corner-3x-scale{scale}.png"), &crop);
        }
      } }
    }
}

/// The glass dividers (`tile_viewer` `dividers::GlassBar`): rods as thick
/// as their rim is wide twice, 16 window pixels, as a 4K split with a zoom
/// strip has them: the split's two closed thirds (each running on past
/// the window's edge) and the strip's divider across the window.
///
/// Over the two panes and the strip, it times four frames: no dividers;
/// the bars alone; the bars over the pictures going on under them as
/// before (each pane's under its half of the gap, the two meeting in its
/// middle, under the strip's divider too); and as now
/// (`dividers::UnderGlass`: the left pane's under all of the gap and the
/// strip's divider up to the gap's far side, the right pane's from there).
#[test]
#[ignore = "GPU cost measurement; run without concurrent GPU tests"]
fn measure_glass_divider_gpu_cost_at_4k() {
    let gpu = Headless::new().expect("Vulkan GPU");
    eprintln!("glass divider benchmark adapter: {:?}", gpu.adapter.get_info());
    let (atlas, world) = fixture(ColorModel::Quantized { bits: 5 });
    let mut renderer = TileRenderer::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm, SHADERS);
    renderer.set_atlas(&gpu.device, &gpu.queue, &atlas);
    renderer.set_world(&gpu.device, &world);
    assert!(renderer.enable_timing(&gpu.device, &gpu.queue));
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("4K glass divider timing"), size: wgpu::Extent3d { width: 3840, height: 2160, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
    });
    const NAMES: [&str; 4] = ["no dividers", "bars alone", "bars over halves (before)", "bars over one view (now)"];
    for scale in [1, 2] {
        let s = scale as i32;
        let t = 16 / s;
        let bar = |[x, y, w, h]: [i32; 4]| Camera { origin: (x, y), cells: false,
            effect: Some(Lens { radius: t as f32 / 2.0, rim: t as f32 / 2.0, glass: true }.into()),
            target_clip: Some([x, y.max(0), x + w, y + h]), ..Camera::new((w as u32, h as u32)) };
        let (w, h) = (3840 / s, 2160 / s);
        let strip = h / 5;
        let along = h - strip - t;
        let bars = [bar([w / 2 - t / 2, strip + t / 2, t, along / 3]), bar([w / 2 - t / 2, strip + t + along * 2 / 3, t, along / 3 + t]),
            bar([-t, strip, w + 2 * t, t])];
        // The panes' world pictures (two places in it) and the strip's.
        let (g0, g1) = (w / 2 - t / 2, w / 2 - t / 2 + t);
        let picture = |x: i32, [rx, ry, rw, rh]: [i32; 4]| Camera { x: x + rx, y: ry, origin: (rx, ry), ..Camera::new((rw.max(1) as u32, rh.max(1) as u32)) };
        let (a, b) = (0, 700);
        let panes = [picture(a, [0, strip + t, g0, along]), picture(b - g1, [g1, strip + t, w - g1, along]),
            Camera { zoom: (2, 1), origin: (0, 0), ..Camera::new((w as u32, strip as u32)) }];
        let middle = (g0 + g1) / 2;
        let halves = [picture(a, [g0, strip + t, middle - g0, along]), picture(b - g1, [middle, strip + t, g1 - middle, along]),
            picture(a, [0, strip, middle, t]), picture(b - g1, [middle, strip, w - middle, t])];
        let one_view = [picture(a, [g0, strip + t, g1 - g0, along]), picture(a, [0, strip, g1, t]), picture(b - g1, [g1, strip, w - g1, t])];
        let configs: [Vec<Camera>; 4] = [panes.to_vec(), [&panes[..], &bars].concat(), [&panes[..], &halves, &bars].concat(), [&panes[..], &one_view, &bars].concat()];
        let mut spans: [Vec<f64>; 4] = Default::default();
        // (Each against the frame without dividers drawn right before or after it.)
        let mut increments: [Vec<f64>; 4] = Default::default();
        for frame in 0..64u32 {
            let mut times_of = [None; 4];
            let order: Vec<usize> = if frame % 2 == 0 { vec![0, 1, 2, 3] } else { vec![3, 2, 1, 0] };
            for k in order {
                let present = tile_render::Present { canvas: (w as u32, h as u32), target: (3840, 2160), zoom: (scale, 1), at: (0, 0) };
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                renderer.render_presented(&gpu.device, &gpu.queue, &mut encoder, &target, &configs[k], &[], frame, [0,0,0,255], present);
                renderer.timer().unwrap().finish(&mut encoder);
                gpu.queue.submit([encoder.finish()]);
                renderer.controls_submitted();
                renderer.timer().unwrap().submitted();
                gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                for times in renderer.timer().unwrap().collect(&gpu.device) {
                    if frame >= 8 {
                        spans[k].push(times.span);
                        times_of[k] = Some(times.span);
                    }
                }
            }
            if let Some(base) = times_of[0] {
                for k in 1..4 {
                    if let Some(t) = times_of[k] { increments[k].push(t - base); }
                }
            }
        }
        let median = |v: &mut Vec<f64>| { v.sort_by(f64::total_cmp); v[v.len() / 2] };
        for k in 0..4 {
            let frame = median(&mut spans[k]);
            if k == 0 {
                eprintln!("4K scale {scale}, {}: frame median {frame:.4} ms, {} samples", NAMES[k], spans[k].len());
            } else {
                let inc = median(&mut increments[k]);
                eprintln!("4K scale {scale}, {}: frame median {frame:.4} ms, paired increment median {inc:.4} ms (p10/p90 {:.4}/{:.4}), {} pairs",
                    NAMES[k], increments[k][increments[k].len() / 10], increments[k][increments[k].len() * 9 / 10], increments[k].len());
            }
        }
    }
}

/// The viewer's shader module (it holds the lens's shaders), drawing with the lens.
const SHADERS: tile_render::Shaders = tile_render::Shaders::new(viewer_spirv::SPIRV, viewer_spirv::SPIRV_FALLBACK).with_effect(glass::effect);

/// The oracle of a renderer drawing with the lens.
const REFERENCE: tile_render::reference::Oracle<'static> = tile_render::reference::Oracle { effect: Some(&glass::ORACLE) };
