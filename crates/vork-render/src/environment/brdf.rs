//! Qualify startup bands using GPU execution time, excluding host completion latency.
use super::*;

const MIN_ROWS: u32 = 8;
const HEADROOM: f64 = 2.;
pub(super) const CACHE_VERSION: u32 = 3;

type Encode<'a> = dyn FnMut(&mut wgpu::CommandEncoder, std::ops::Range<u32>, Option<wgpu::ComputePassTimestampWrites<'_>>) + 'a;

fn next_rows(rows: u32, elapsed_ms: f64, height: u32) -> u32 {
    if !elapsed_ms.is_finite() || elapsed_ms <= 0. { return rows; }
    // A linear upper estimate can qualify a full-region candidate directly.
    // Otherwise grow gradually: tiny grids can underutilize the device badly.
    if height as f64 / rows as f64 * elapsed_ms * HEADROOM <= crate::constants::SAMPLE_GROUP_GPU_MS {
        return height;
    }
    let growth = (crate::constants::SAMPLE_GROUP_GPU_MS / elapsed_ms)
        .floor().clamp(1., 4.) as u32;
    rows.saturating_mul(growth).min(height)
}

pub(super) fn initialize(
    r: &Renderer,
    mut encode: impl FnMut(&mut wgpu::CommandEncoder, std::ops::Range<u32>, Option<wgpu::ComputePassTimestampWrites<'_>>),
) -> Result<(wgpu::SubmissionIndex, [u32; 2])> {
    let regions = [0..crate::constants::BRDF_SIZE, crate::constants::BRDF_SIZE..crate::constants::BRDF_HEIGHT];
    let path = r.disk_cache.path("brdf-band-rows", CACHE_VERSION, blake3::hash(FILTER_SHADER.as_bytes()));
    let cached = disk_cache::read(&path)
        .and_then(|bytes| serde_json::from_slice::<[u32; 2]>(&bytes).ok())
        .filter(|rows| rows.iter().zip(&regions).all(|(rows, range)| {
            *rows >= MIN_ROWS && *rows <= range.end - range.start && rows % MIN_ROWS == 0
        }));
    if let Some(rows) = cached {
        let last = publish(r, &mut encode, rows)?;
        return Ok((last, rows));
    }
    // Moment rows contain one long serial hemispherical quadrature each.
    // Even an eight-row grid pays nearly its full latency on parallel GPUs;
    // repeated endpoint probes mostly repeat that work. Start with the bounded
    // 128-row moment region, never the full 2240-row atlas. Dielectric work
    // scales with rows and must still start small on watchdog-sensitive GPUs.
    let mut rows = [crate::constants::BRDF_SIZE, MIN_ROWS];
    let mut qualified = [MIN_ROWS; 2];
    let mut active = [true; 2];
    let mut completed = [false; 2];
    let mut last;
    loop {
        // Probe both ends together, with one readback fence. Each probe still
        // owns a separate native command buffer on watchdog-sensitive devices.
        let mut bands = Vec::new();
        for i in 0..2 {
            if !active[i] { continue; }
            let range = &regions[i];
            bands.push((i, range.start..range.start + rows[i]));
            if rows[i] < range.end - range.start {
                bands.push((i, range.end - rows[i]..range.end));
            }
        }
        let (submission, mut costs) = measure(r, &mut encode, &bands)?;
        last = submission;
        // A first-dispatch clock/compiler penalty must not permanently freeze
        // a tiny schedule. Retry only stalled initial candidates, once.
        let stalled: Vec<_> = bands.iter().filter(|(i, _)| {
            rows[*i] == MIN_ROWS && next_rows(rows[*i], costs[*i], regions[*i].end - regions[*i].start) == MIN_ROWS
        }).cloned().collect();
        if !stalled.is_empty() {
            let (submission, retry) = measure(r, &mut encode, &stalled)?;
            last = submission;
            for (i, _) in &stalled { costs[*i] = costs[*i].min(retry[*i]); }
        }
        for i in 0..2 {
            if !active[i] { continue; }
            let height = regions[i].end - regions[i].start;
            if costs[i] * HEADROOM > crate::constants::SAMPLE_GROUP_GPU_MS && rows[i] > MIN_ROWS {
                rows[i] = qualified[i];
                active[i] = false;
                continue;
            }
            qualified[i] = rows[i];
            completed[i] = rows[i] == height;
            let next = next_rows(rows[i], costs[i], height);
            active[i] = next != rows[i];
            rows[i] = next;
        }
        if !active.into_iter().any(|value| value) { break; }
    }
    // A full-region qualifying probe already wrote every texel. Do not repeat it.
    for (i, range) in regions.into_iter().enumerate() {
        if completed[i] { continue; }
        for first in (range.start..range.end).step_by(rows[i] as usize) {
            let mut encoder = r.device.create_command_encoder(&Default::default());
            encode(&mut encoder, first..(first + rows[i]).min(range.end), None);
            last = crate::command_finish::Segments::submit(&r.queue, encoder);
        }
    }
    r.wait(Some(last.clone()))?;
    if let Ok(bytes) = serde_json::to_vec(&rows) { let _ = disk_cache::write(&path, &bytes); }
    Ok((last, rows))
}

fn publish(r: &Renderer, encode: &mut Encode<'_>, rows: [u32; 2]) -> Result<wgpu::SubmissionIndex> {
    if rows == [crate::constants::BRDF_SIZE, crate::constants::BRDF_HEIGHT - crate::constants::BRDF_SIZE] {
        // Fast devices keep the original one-dispatch, one-submission warm path.
        let mut encoder = r.device.create_command_encoder(&Default::default());
        encode(&mut encoder, 0..crate::constants::BRDF_HEIGHT, None);
        return Ok(crate::command_finish::Segments::submit(&r.queue, encoder));
    }
    let mut last = None;
    for (i, range) in [0..crate::constants::BRDF_SIZE, crate::constants::BRDF_SIZE..crate::constants::BRDF_HEIGHT].into_iter().enumerate() {
        for first in (range.start..range.end).step_by(rows[i] as usize) {
            let mut encoder = r.device.create_command_encoder(&Default::default());
            encode(&mut encoder, first..(first + rows[i]).min(range.end), None);
            last = Some(crate::command_finish::Segments::submit(&r.queue, encoder));
        }
    }
    Ok(last.expect("nonempty BRDF atlas"))
}

fn measure(r: &Renderer, encode: &mut Encode<'_>, bands: &[(usize, std::ops::Range<u32>)]) -> Result<(wgpu::SubmissionIndex, [f64; 2])> {
    let count = bands.len() as u32 * 2;
    let queries = r.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("BRDF startup costs"), ty: wgpu::QueryType::Timestamp, count,
    });
    let resolved = crate::gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
        label: Some("BRDF cost resolve"), size: wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let read = crate::gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
        label: Some("BRDF cost readback"), size: u64::from(count) * 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    let start = std::time::Instant::now();
    let mut commands = Vec::new();
    for (index, (_, range)) in bands.iter().enumerate() {
        let mut encoder = r.device.create_command_encoder(&Default::default());
        encode(&mut encoder, range.clone(), Some(wgpu::ComputePassTimestampWrites {
            query_set: &queries, beginning_of_pass_write_index: Some(index as u32 * 2), end_of_pass_write_index: Some(index as u32 * 2 + 1),
        }));
        commands.push(encoder.finish());
    }
    let mut encoder = r.device.create_command_encoder(&Default::default());
    encoder.resolve_query_set(&queries, 0..count, &resolved, 0);
    encoder.copy_buffer_to_buffer(&resolved, 0, &read, 0, u64::from(count) * 8);
    let (tx, rx) = std::sync::mpsc::channel();
    encoder.map_buffer_on_submit(&read, wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
    commands.push(encoder.finish());
    let submission = r.queue.submit(commands);
    r.wait(Some(submission.clone()))?;
    rx.recv()??;
    let host_ms = start.elapsed().as_secs_f64() * 1000.;
    let mapped = read.get_mapped_range(..);
    let ticks: &[u64] = bytemuck::cast_slice(&mapped);
    let mut costs = [0f64; 2];
    for (index, (region, _)) in bands.iter().enumerate() {
        // Broken or zero counters never imply free work. The completed host
        // envelope remains a conservative fallback, without growing on it.
        let cost = crate::timing::elapsed_ms(ticks[index * 2], ticks[index * 2 + 1], r.queue.get_timestamp_period())
            .filter(|ms| ms.is_finite() && *ms > 0.).unwrap_or(host_ms);
        costs[*region] = costs[*region].max(cost);
    }
    #[cfg(test)]
    eprintln!("BRDF probes={bands:?} gpu_ms={costs:?} host_ms={host_ms:.3}");
    Ok((submission, costs))
}

#[test]
fn measured_bands_keep_alignment_and_timing_headroom() {
    for invalid in [0., -1., f64::NAN, f64::INFINITY] { assert_eq!(next_rows(8, invalid, 128), 8); }
    // A constant dispatch latency is not linear work: measure the larger
    // candidate within the budget, rather than applying headroom twice.
    assert_eq!(next_rows(8, 60., 128), 32);
    assert_eq!(next_rows(32, 60., 128), 128);
    assert_eq!(next_rows(8, 300., 128), 8);
    for rows in [8, 32, 128] {
        for cost in [0.001, 0.1, 1., 10., 100., 1000.] {
            let next = next_rows(rows, cost, 128);
            assert_eq!(next % 8, 0);
            assert!((rows..=128).contains(&next));
            assert!(next == 128 || next <= rows * 4);
            assert!(next == rows || next as f64 / rows as f64 * cost <= crate::constants::SAMPLE_GROUP_GPU_MS);
        }
    }
}
