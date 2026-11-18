use super::*;

fn read_lookup_bytes(r: &Renderer, texture: &wgpu::Texture) -> Result<Vec<u8>> {
    let stride = (texture.width() * 8).div_ceil(256) * 256;
    let read = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BRDF lookup regression readback"),
        size: u64::from(stride) * u64::from(texture.height()),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(texture.as_image_copy(), wgpu::TexelCopyBufferInfo {
        buffer: &read,
        layout: wgpu::TexelCopyBufferLayout {
            offset: 0, bytes_per_row: Some(stride), rows_per_image: Some(texture.height()),
        },
    }, texture.size());
    let submission = r.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    read.map_async(wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
    r.wait(Some(submission))?;
    rx.recv()??;
    let mapped = read.get_mapped_range(..);
    Ok(mapped.chunks_exact(stride as usize)
        .flat_map(|row| row[..texture.width() as usize * 8].iter().copied()).collect())
}

fn read_lookup(r: &Renderer, texture: &wgpu::Texture) -> Result<Vec<[f32; 4]>> {
    Ok(read_lookup_bytes(r, texture)?.chunks_exact(8).map(|pixel| {
        std::array::from_fn(|channel| {
            half::f16::from_bits(u16::from_le_bytes(pixel[channel * 2..channel * 2 + 2].try_into().unwrap())).to_f32()
        })
    }).collect())
}

#[test]
#[ignore = "requires real GPU and pinned Slang"]
fn startup_brdf_lookup_preserves_moments_and_dielectric_atlas() -> Result<()> {
    let r = Renderer::new()?;
    let scene = vork_scene::v2::from_test_value(serde_json::json!({
        "version": 2, "nodes": [{"type": "camera"}], "render": {"resolution": [16, 16]},
    }))?;
    let installed = r.install(scene, Path::new("."))?;
    let lookup = installed.environment.brdf.texture();
    let actual = read_lookup(&r, lookup)?;
    let width = lookup.width() as usize;

    // An untiled prefix spans two eight-row boundaries and a partial final
    // workgroup. It uses zero offset, independently of the startup row schedule.
    let reference = r.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("untiled BRDF prefix"),
        size: wgpu::Extent3d { width: lookup.width(), height: 17, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let parameters = filter_parameters(0, 2, 1., [glam::DVec3::ZERO; 4], [1, 1]);
    let params = init(&r.device, "untiled BRDF parameters", bytemuck::cast_slice(&parameters), wgpu::BufferUsages::UNIFORM);
    let cdf = init(&r.device, "unused BRDF proposal", bytemuck::cast_slice(&[1f32, 1.]), wgpu::BufferUsages::STORAGE);
    let view = reference.create_view(&Default::default());
    let pipeline = Environment::pipeline(&r);
    let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("untiled BRDF reference"), layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            entry(0, wgpu::BindingResource::TextureView(&installed.environment.source)),
            entry(1, wgpu::BindingResource::TextureView(&view)),
            entry(2, params.as_entire_binding()), entry(3, cdf.as_entire_binding()),
        ],
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(lookup.width().div_ceil(8), 3, 1);
    }
    r.queue.submit([encoder.finish()]);
    let expected = read_lookup(&r, &reference)?;
    assert_eq!(&actual[..expected.len()], expected.as_slice(), "bounded startup changed BRDF moments across row boundaries");

    // Every valid dielectric texel writes alpha=1, including the zero-energy
    // eta=1 slice. Check every row to catch skipped or overwritten tiles.
    let n = vork_slang::registry::BRDF_ALBEDO_SIZE as usize;
    let first = vork_slang::registry::BRDF_SIZE as usize;
    for y in first..lookup.height() as usize {
        for x in 0..width {
            let pixel = actual[y * width + x];
            let valid = ((y - first) / n) * 2 + x / n <= vork_slang::registry::BRDF_IOR_STEPS as usize;
            assert!(pixel.iter().all(|v| v.is_finite()), "nonfinite BRDF texel ({x}, {y}): {pixel:?}");
            assert_eq!(pixel[3], if valid { 1. } else { 0. }, "unwritten dielectric texel ({x}, {y})");
        }
    }
    // Independent physical references: smooth normal-incidence Fresnel, equal
    // IOR (no interface), and unit-Fresnel GGX at alpha=1 (1 - ln(2)).
    for (x, y, expected) in [
        (63, 128, 1.), (127, 128, 0.9384765625), (63, 192, 0.87890625),
        (127, 1088, 0.0009765625), (63, 1152, 0.), (127, 2112, 0.9384765625),
        (63, 2176, 1.), (63, 2239, 1. - std::f32::consts::LN_2),
    ] {
        let pixel = actual[y * width + x];
        assert!((pixel[0] - expected).abs() < 0.001, "dielectric albedo ({x}, {y}): {pixel:?}, expected {expected}");
        assert_eq!([pixel[1], pixel[2]], [0., 0.]);
    }
    Ok(())
}

// The full untiled reference is intentionally opt-in: it exceeds the watchdog
// budget on affected M2 Pro devices. Run on a qualified Metal GPU or hardware Vulkan.
#[test]
#[ignore = "requires real GPU capable of the original whole-atlas dispatch"]
fn startup_brdf_lookup_bytes_and_timing() -> Result<()> {
    let r = Renderer::new()?;
    eprintln!("BRDF adapter: {:?}", r.adapter_info);
    let scene = vork_scene::v2::from_test_value(serde_json::json!({
        "version": 2, "nodes": [{"type": "camera"}], "render": {"resolution": [64, 64]},
    }))?;
    let installed = r.install(scene, Path::new("."))?;
    let lookup = installed.environment.brdf.texture();
    let actual = read_lookup_bytes(&r, lookup)?;
    let reference = r.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("BRDF schedule comparison"), size: lookup.size(),
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = reference.create_view(&Default::default());
    let cdf = init(&r.device, "unused BRDF proposal", bytemuck::cast_slice(&[1f32, 1.]), wgpu::BufferUsages::STORAGE);
    let pipeline = Environment::pipeline(&r);
    // Exercise cold calibration once, then cached schedules in every warm pair.
    let costs_path = r.disk_cache.path("brdf-band-rows", brdf::CACHE_VERSION, blake3::hash(FILTER_SHADER.as_bytes()));
    if costs_path.exists() { std::fs::remove_file(&costs_path)?; }
    // Alternate order, discarding the first pair to avoid pipeline/clock warmup.
    for round in 0..8 {
        for rows in if round % 2 == 0 { [lookup.height(), 0] } else { [0, lookup.height()] } {
            let start = std::time::Instant::now();
            let mut encoded_ranges = Vec::new();
            let mut encode = |encoder: &mut wgpu::CommandEncoder, range: std::ops::Range<u32>, timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>| {
                encoded_ranges.push(range.clone());
                let mut parameters = filter_parameters(0, 2, 1., [glam::DVec3::ZERO; 4], [1, 1]);
                parameters[5][3] = range.start as f32;
                let params = init(&r.device, "BRDF schedule parameters", bytemuck::cast_slice(&parameters), wgpu::BufferUsages::UNIFORM);
                let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("BRDF schedule comparison"), layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        entry(0, wgpu::BindingResource::TextureView(&installed.environment.source)),
                        entry(1, wgpu::BindingResource::TextureView(&view)),
                        entry(2, params.as_entire_binding()), entry(3, cdf.as_entire_binding()),
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label: None, timestamp_writes});
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(lookup.width().div_ceil(8), (range.end - range.start).div_ceil(8), 1);
            };
            let submission = if rows == 0 {
                let before = crate::command_finish::submission_count();
                let (submission, selected) = brdf::initialize(&r, &mut encode)?;
                if round == 0 && selected[0] == crate::constants::BRDF_SIZE {
                    let moment_ranges: Vec<_> = encoded_ranges.iter()
                        .filter(|range| range.start < crate::constants::BRDF_SIZE).cloned().collect();
                    assert_eq!(moment_ranges, [0..crate::constants::BRDF_SIZE],
                        "cold qualification must produce the complete moment region only once");
                }
                if round > 0 && selected == [crate::constants::BRDF_SIZE, crate::constants::BRDF_HEIGHT - crate::constants::BRDF_SIZE] {
                    assert_eq!(crate::command_finish::submission_count() - before, 1,
                        "qualified whole-atlas startup must keep the original single submission");
                }
                eprintln!("BRDF calibrated_rows={selected:?}");
                submission
            } else {
                let mut encoder = r.device.create_command_encoder(&Default::default());
                encode(&mut encoder, 0..lookup.height(), None);
                crate::command_finish::Segments::submit(&r.queue, encoder)
            };
            r.wait(Some(submission))?;
            eprintln!("BRDF round={round} rows={rows} elapsed_ms={:.3}", start.elapsed().as_secs_f64() * 1000.);
            let expected = read_lookup_bytes(&r, &reference)?;
            assert_eq!(actual, expected, "full atlas bytes changed with rows={rows}");
            if let Some(dir) = std::env::var_os("VORK_BRDF_EVIDENCE") {
                let dir = std::path::PathBuf::from(dir);
                std::fs::create_dir_all(&dir)?;
                std::fs::write(dir.join("startup.rgba16f"), &actual)?;
                std::fs::write(dir.join(format!("rows-{rows}.rgba16f")), &expected)?;
            }
        }
    }
    eprintln!("BRDF identical full atlas: {} bytes", actual.len());
    Ok(())
}
