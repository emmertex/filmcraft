//! CDEF (7.15).

use crate::frame::{FrameBuf, RowBand};
use crate::par::Pool;
use crate::postfilter::bands;
use crate::spec_tables::*;
use crate::state::FrameState;

/// Run CDEF on `fs.cur`; returns the CdefFrame, or None when no block is filtered (CdefFrame
/// equals CurrFrame). Bands of 64 luma rows (whole 8x8 blocks) are filtered in parallel: every
/// block reads only CurrFrame and writes only its own samples.
pub(crate) fn apply(fs: &FrameState, pool: &Pool) -> Option<FrameBuf> {
    let fh = &fs.fh;
    if fh.coded_lossless || fh.allow_intrabc || !fs.seq.enable_cdef {
        return None;
    }
    let (mi_rows, mi_cols) = (fh.mi_rows as usize, fh.mi_cols as usize);
    let mut out = fs.cur.clone();
    let ranges = bands(pool, mi_rows * 4);
    let mut planes = out.row_bands(&ranges);
    let any = pool.map_mut(&mut planes, |b, band| {
        let (start, end) = ranges[b];
        let mut any = false;
        let mut r = start / 4;
        while r < mi_rows.min(end / 4) {
            let mut c = 0;
            while c < mi_cols {
                let idx = fs.cdef_idx(r & !15, c & !15);
                if idx >= 0 && cdef_block(fs, band, r, c, idx as usize) {
                    any = true;
                }
                c += 2;
            }
            r += 2;
        }
        any
    });
    drop(planes);
    any.contains(&true).then_some(out)
}

fn cdef_block(fs: &FrameState, out: &mut [RowBand; 3], r: usize, c: usize, idx: usize) -> bool {
    let mi = &fs.mi;
    let sk = |rr: usize, cc: usize| -> bool { if rr < mi.rows && cc < mi.cols { mi.skip[mi.idx(rr, cc)] } else { true } };
    let skip = sk(r, c) && sk(r + 1, c) && sk(r, c + 1) && sk(r + 1, c + 1);
    if skip {
        return false;
    }
    let coeff_shift = fs.bit_depth - 8;
    let (y_dir, var) = direction(fs, r, c);
    let cp = &fs.fh.cdef;
    let mut pri = (cp.y_pri[idx] << coeff_shift) as i32;
    let sec = (cp.y_sec[idx] << coeff_shift) as i32;
    let dir = if pri == 0 { 0 } else { y_dir };
    let var_str = if (var >> 6) != 0 { (31 - ((var >> 6) as u32).leading_zeros()).min(12) as i32 } else { 0 };
    pri = if var != 0 { (pri * (4 + var_str) + 8) >> 4 } else { 0 };
    let damping = cp.damping as i32 + coeff_shift as i32;
    filter(fs, out, 0, r, c, pri, sec, damping, dir);
    if fs.num_planes == 1 {
        return true;
    }
    let pri = (cp.uv_pri[idx] << coeff_shift) as i32;
    let sec = (cp.uv_sec[idx] << coeff_shift) as i32;
    let dir = if pri == 0 { 0 } else { CDEF_UV_DIR[fs.ssx][fs.ssy][y_dir] as usize };
    let damping = cp.damping as i32 + coeff_shift as i32 - 1;
    if fs.ssx == 1 && fs.ssy == 1 && filter_uv420(fs, out, r, c, pri, sec, damping, dir) {
        return true;
    }
    filter(fs, out, 1, r, c, pri, sec, damping, dir);
    filter(fs, out, 2, r, c, pri, sec, damping, dir);
    true
}

fn direction(fs: &FrameState, r: usize, c: usize) -> (usize, i32) {
    let mut cost = [0i32; 8];
    // Each row's 8 samples (and their pair sums) are added to every direction's line sums as
    // whole shifted vectors, which vectorises.
    let mut partial = [[0i32; 16]; 8];
    let x0 = c << 2;
    let y0 = r << 2;
    let pl = &fs.cur.planes[0];
    let sh = fs.bit_depth - 8;
    for i in 0..8 {
        let Some(row) = pl.row(y0 + i).get(x0..).and_then(|r| r.first_chunk::<8>()) else {
            continue;
        };
        let x: [i32; 8] = std::array::from_fn(|j| (row[j] as i32 >> sh) - 128);
        let rev: [i32; 8] = std::array::from_fn(|j| x[7 - j]);
        let pair: [i32; 4] = std::array::from_fn(|k| x[2 * k] + x[2 * k + 1]);
        let add = |dst: &mut [i32; 16], at: usize, v: &[i32]| {
            for (d, &v) in dst[at..at + v.len()].iter_mut().zip(v) {
                *d += v;
            }
        };
        add(&mut partial[0], i, &x);
        add(&mut partial[1], i, &pair);
        partial[2][i] = x.iter().sum();
        // partial[3][3 + i - k] += pair[k]
        let prev: [i32; 4] = std::array::from_fn(|k| pair[3 - k]);
        add(&mut partial[3], i, &prev);
        // partial[4][7 + i - j] += x[j]
        add(&mut partial[4], i, &rev);
        add(&mut partial[5], 3 - i / 2, &x);
        add(&mut partial[6], 0, &x);
        add(&mut partial[7], i / 2, &x);
    }
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= DIV_TABLE[8] as i32;
    cost[6] *= DIV_TABLE[8] as i32;
    for i in 0..7 {
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i]) * DIV_TABLE[i + 1] as i32;
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i]) * DIV_TABLE[i + 1] as i32;
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV_TABLE[8] as i32;
    cost[4] += partial[4][7] * partial[4][7] * DIV_TABLE[8] as i32;
    let mut i = 1;
    while i < 8 {
        for j in 0..5 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= DIV_TABLE[8] as i32;
        for j in 0..3 {
            cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j]) * DIV_TABLE[2 * j + 2] as i32;
        }
        i += 2;
    }
    let mut best = 0;
    let mut y_dir = 0;
    for (i, &c) in cost.iter().enumerate() {
        if c > best {
            best = c;
            y_dir = i;
        }
    }
    (y_dir, (best - cost[(y_dir + 4) & 7]) >> 10)
}

/// Marks a sample outside the frame (CdefAvailable = 0) in the padded block.
const CDEF_NA: i32 = i32::MIN;
/// Padded block: 8x8 plus 2 samples on every side, row stride 12.
const PAD_STRIDE: usize = 12;

#[inline(always)]
fn constrain(diff: i32, threshold: i32, adj: u32) -> i32 {
    let mag = (threshold - (diff.abs() >> adj)).clamp(0, diff.abs());
    if diff < 0 { -mag } else { mag }
}

#[allow(clippy::too_many_arguments)]
/// Tap displacements in the padded block: [primary k = 0, 1], then the two secondary
/// directions for k = 0 and k = 1.
fn tap_offsets(dir: usize) -> [[isize; 2]; 3] {
    let off = |d: usize, k: usize| CDEF_DIRECTIONS[d][k][0] as isize * PAD_STRIDE as isize + CDEF_DIRECTIONS[d][k][1] as isize;
    let d_lo = (dir + 6) & 7;
    let d_hi = (dir + 2) & 7;
    [[off(dir, 0), off(dir, 1)], [off(d_lo, 0), off(d_hi, 0)], [off(d_lo, 1), off(d_hi, 1)]]
}

/// Tap parameters (displacement, weight, strength, damping shift) shared by `filter_rows` and
/// the per-sample path.
fn tap_params(fs: &FrameState, pri: i32, sec: i32, damping: i32) -> ([i32; 2], [i32; 2], u32, u32) {
    let pri_tap = ((pri >> (fs.bit_depth - 8)) & 1) as usize;
    let pri_taps = [CDEF_PRI_TAPS[pri_tap][0] as i32, CDEF_PRI_TAPS[pri_tap][1] as i32];
    let sec_taps = [CDEF_SEC_TAPS[pri_tap][0] as i32, CDEF_SEC_TAPS[pri_tap][1] as i32];
    let adj = |t: i32| if t == 0 { 0 } else { (damping - (31 - (t as u32).leading_zeros()) as i32).max(0) as u32 };
    (pri_taps, sec_taps, adj(pri), adj(sec))
}

type Taps = [(isize, i32, i32, u32); 12];

fn make_taps(fs: &FrameState, pri: i32, sec: i32, damping: i32, dir: usize) -> Taps {
    let [pri_off, sec0, sec1] = tap_offsets(dir);
    let sec_off = [sec0, sec1];
    let (pri_taps, sec_taps, pri_adj, sec_adj) = tap_params(fs, pri, sec, damping);
    let mut taps = [(0isize, 0i32, 0i32, 0u32); 12];
    let mut n = 0;
    for k in 0..2 {
        for sign in [-1isize, 1] {
            taps[n] = (sign * pri_off[k], pri_taps[k], pri, pri_adj);
            taps[n + 1] = (sign * sec_off[k][0], sec_taps[k], sec, sec_adj);
            taps[n + 2] = (sign * sec_off[k][1], sec_taps[k], sec, sec_adj);
            n += 3;
        }
    }
    taps
}

/// Both 4:2:0 chroma planes of an interior 8x8 luma block in one pass (they share strengths
/// and direction): each 8-lane row holds U's 4 samples then V's. Returns false (nothing done)
/// when the block is not interior.
fn filter_uv420(fs: &FrameState, out: &mut [RowBand; 3], r: usize, c: usize, pri: i32, sec: i32, damping: i32, dir: usize) -> bool {
    let (x0, y0) = (c * 2, r * 2);
    let (mi_rows, mi_cols) = (fs.fh.mi_rows as usize, fs.fh.mi_cols as usize);
    if !(y0 >= 2 && x0 >= 2 && ((y0 + 5) << 1) >> 2 < mi_rows && ((x0 + 5) << 1) >> 2 < mi_cols) {
        return false;
    }
    if pri == 0 && sec == 0 {
        return true;
    }
    const S: usize = 8;
    let mut b = [[0i16; S * S]; 2];
    for (p, bp) in b.iter_mut().enumerate() {
        let src = &fs.cur.planes[1 + p];
        for i in 0..S {
            let row = &src.row(y0 + i - 2)[x0 - 2..x0 + 6];
            for (d, &v) in bp[i * S..i * S + S].iter_mut().zip(row) {
                *d = v as i16;
            }
        }
    }
    // Displacements in the 8-wide padded block (make_taps uses PAD_STRIDE = 12).
    let taps = make_taps(fs, pri, sec, damping, dir).map(|(off, w, t, a)| {
        let (dy, dx) = ((off + 2 * PAD_STRIDE as isize + 2).div_euclid(PAD_STRIDE as isize) - 2, (off + 2 * PAD_STRIDE as isize + 2).rem_euclid(PAD_STRIDE as isize) - 2);
        (dy * S as isize + dx, w, t, a)
    });
    // g[0]: the centre samples, g[1 + t]: tap t, each as 32 lanes (U rows 0..4, then V rows
    // 0..4, 4 samples each), gathered once so the lane loops run over contiguous memory.
    let mut g = [[0i16; 32]; 13];
    let offs = std::iter::once(0isize).chain(taps.iter().map(|t| t.0));
    for (gt, off) in g.iter_mut().zip(offs) {
        for (p, bp) in b.iter().enumerate() {
            for i in 0..4 {
                let at = (((i + 2) * S + 2) as isize + off) as usize;
                if let (Some(src), Some(dst)) = (bp.get(at..at + 4), gt.get_mut(p * 16 + i * 4..p * 16 + i * 4 + 4)) {
                    dst.copy_from_slice(src);
                }
            }
        }
    }
    let [x, rest @ ..] = &g;
    let (mut sum, mut max, mut min) = ([0i16; 32], *x, *x);
    for (p, &(_, weight, thr, adj)) in rest.iter().zip(&taps) {
        tap(p, x, &mut sum, &mut max, &mut min, weight as i16, thr as i16, adj);
    }
    let mut o = [0u16; 32];
    for j in 0..32 {
        let s = sum[j];
        o[j] = (x[j] + ((8 + s - (s < 0) as i16) >> 4)).max(min[j]).min(max[j]) as u16;
    }
    for p in 0..2 {
        for i in 0..4 {
            out[1 + p].row_mut(y0 + i)[x0..x0 + 4].copy_from_slice(&o[p * 16 + i * 4..p * 16 + i * 4 + 4]);
        }
    }
    true
}

fn filter(fs: &FrameState, out: &mut [RowBand; 3], plane: usize, r: usize, c: usize, pri: i32, sec: i32, damping: i32, dir: usize) {
    if pri == 0 && sec == 0 {
        // Every tap contributes 0 and the clamp to [min, max] keeps x: CdefFrame = CurrFrame.
        return;
    }
    let (sub_x, sub_y) = if plane > 0 { (fs.ssx, fs.ssy) } else { (0, 0) };
    let x0 = (c * 4) >> sub_x;
    let y0 = (r * 4) >> sub_y;
    let w = 8 >> sub_x;
    let h = 8 >> sub_y;
    let src = &fs.cur.planes[plane];
    let (mi_rows, mi_cols) = (fs.fh.mi_rows as isize, fs.fh.mi_cols as isize);
    // Interior blocks (the 2-sample border lies inside the frame's 4x4 blocks) take every tap:
    // copy whole rows straight into the i16 block.
    let interior = y0 >= 2 && x0 >= 2 && (((y0 + h + 1) << sub_y) >> 2) < mi_rows as usize && (((x0 + w + 1) << sub_x) >> 2) < mi_cols as usize;
    let [pri_off, sec0, sec1] = tap_offsets(dir);
    let sec_off = [sec0, sec1];
    let (pri_taps, sec_taps, pri_adj, sec_adj) = tap_params(fs, pri, sec, damping);
    let taps = make_taps(fs, pri, sec, damping, dir);
    let dst = &mut out[plane];
    let mut b16 = [0i16; PAD_STRIDE * PAD_STRIDE];
    // Every tap is available: whole rows at a time (constrain() with threshold 0 is 0).
    let fast = |b16: &[i16; PAD_STRIDE * PAD_STRIDE], dst: &mut RowBand| match w {
        8 => filter_rows::<8>(b16, &taps, dst, (y0, x0, h)),
        _ => filter_rows::<4>(b16, &taps, dst, (y0, x0, h)),
    };
    if interior {
        for i in 0..h + 4 {
            let row = &src.row(y0 + i - 2)[x0 - 2..x0 + w + 2];
            for (d, &s) in b16[i * PAD_STRIDE..].iter_mut().zip(row) {
                *d = s as i16;
            }
        }
        fast(&b16, dst);
        return;
    }
    // CdefAvailable per sample: its 4x4 block is inside the frame.
    let mut buf = [CDEF_NA; PAD_STRIDE * PAD_STRIDE];
    let mut all_avail = true;
    for i in 0..h + 4 {
        let y = (y0 + i) as isize - 2;
        let cand_r = (y << sub_y) >> 2;
        if y < 0 || cand_r >= mi_rows {
            all_avail = false;
            continue;
        }
        let row = src.row(y as usize);
        for j in 0..w + 4 {
            let x = (x0 + j) as isize - 2;
            let cand_c = (x << sub_x) >> 2;
            if x >= 0 && cand_c < mi_cols {
                buf[i * PAD_STRIDE + j] = row[x as usize] as i32;
            } else {
                all_avail = false;
            }
        }
    }
    if all_avail {
        for (d, &s) in b16.iter_mut().zip(buf.iter()) {
            *d = s as i16;
        }
        fast(&b16, dst);
        return;
    }
    for i in 0..h {
        // The centre sample is read even where CdefAvailable is 0 (8x8 blocks straddling an odd
        // MiRows / MiCols edge).
        let srow = &src.row(y0 + i)[x0..x0 + w];
        let drow = &mut dst.row_mut(y0 + i)[x0..x0 + w];
        for j in 0..w {
            let ci = ((i + 2) * PAD_STRIDE + j + 2) as isize;
            let x = srow[j] as i32;
            let mut sum = 0i32;
            let mut max = x;
            let mut min = x;
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    let p = buf[(ci + sign * pri_off[k]) as usize];
                    if p != CDEF_NA {
                        if pri != 0 {
                            sum += pri_taps[k] * constrain(p - x, pri, pri_adj);
                        }
                        max = max.max(p);
                        min = min.min(p);
                    }
                    for s in 0..2 {
                        let q = buf[(ci + sign * sec_off[k][s]) as usize];
                        if q != CDEF_NA {
                            if sec != 0 {
                                sum += sec_taps[k] * constrain(q - x, sec, sec_adj);
                            }
                            max = max.max(q);
                            min = min.min(q);
                        }
                    }
                }
            }
            drow[j] = (x + ((8 + sum - (sum < 0) as i32) >> 4)).clamp(min, max) as u16;
        }
    }
}

/// CDEF of a block whose taps are all available, `W` samples per row. Samples (at most 12
/// bits), differences and the weighted tap sum all fit in i16, which doubles the SIMD width.
#[inline(always)]
fn filter_rows<const W: usize>(b16: &[i16; PAD_STRIDE * PAD_STRIDE], taps: &Taps, dst: &mut RowBand, (y0, x0, h): (usize, usize, usize)) {
    for i in 0..h {
        let ci = ((i + 2) * PAD_STRIDE + 2) as isize;
        let Some(&x) = b16.get(ci as usize..).and_then(|r| r.first_chunk::<W>()) else {
            return;
        };
        let (mut sum, mut max, mut min) = ([0i16; W], x, x);
        for &(off, weight, thr, adj) in taps {
            let Some(p) = b16.get((ci + off) as usize..).and_then(|r| r.first_chunk::<W>()) else {
                return;
            };
            tap(p, &x, &mut sum, &mut max, &mut min, weight as i16, thr as i16, adj);
        }
        let drow = &mut dst.row_mut(y0 + i)[x0..x0 + W];
        for j in 0..W {
            let s = sum[j];
            drow[j] = (x[j] + ((8 + s - (s < 0) as i16) >> 4)).max(min[j]).min(max[j]) as u16;
        }
    }
}

/// One CDEF tap over a block's lanes: constrain() written with selects and a branch-free sign,
/// so the loop vectorises.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tap(p: &[i16], x: &[i16], sum: &mut [i16], max: &mut [i16], min: &mut [i16], weight: i16, thr: i16, adj: u32) {
    let adj = (adj & 15) as i16;
    let n = p.len();
    let (x, sum, max, min) = (&x[..n], &mut sum[..n], &mut max[..n], &mut min[..n]);
    for j in 0..n {
        let d = p[j].wrapping_sub(x[j]);
        let sign = d >> 15;
        let ad = (d ^ sign) - sign;
        let m = thr - (ad >> adj);
        let m = if m < 0 { 0 } else { m };
        let mag = if m < ad { m } else { ad };
        sum[j] = sum[j].wrapping_add(weight.wrapping_mul((mag ^ sign) - sign));
        max[j] = if p[j] > max[j] { p[j] } else { max[j] };
        min[j] = if p[j] < min[j] { p[j] } else { min[j] };
    }
}
