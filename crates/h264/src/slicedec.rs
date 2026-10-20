//! Slice data decoding: macroblock parsing (CAVLC), motion vector derivation, and reconstruction.

use crate::cavlc;
use crate::deblock;
use crate::error::{Result, ensure, invalid};
use crate::inter::{self, Weight};
use crate::intra::{self, Avail};
use crate::mbtypes::*;
use crate::params::{Pps, Sps};
use crate::picture::*;
use crate::slice::{SliceHeader, SliceType};
use crate::tables::*;
use crate::transform::{self, LevelScale};
use filmcraft_bitstream::BitReader;
use std::sync::Arc;

/// Per-slice data kept with the picture (deblocking, motion field export).
#[derive(Clone, Debug, Default)]
pub struct SliceInfo {
    pub disable_deblocking_filter_idc: u32,
    pub alpha_offset: i32,
    pub beta_offset: i32,
    /// Unique ids of the pictures in RefPicList0/1.
    pub ref_ids: [Vec<u32>; 2],
}

/// The picture under construction: private reconstruction buffers plus the shared [`Frame`] that
/// finished (deblocked) macroblock rows are published into.
///
/// A field picture of an interlaced stream has its own (half-height) buffers and macroblock grid;
/// frame pictures of interlaced streams publish pairs of macroblock rows (one field row each).
pub struct PicState {
    pub planes: Planes,
    pub mbs: Vec<MbState>,
    pub slices: Vec<SliceInfo>,
    /// Macroblock grid of the coded picture (rows of one field for a field picture).
    pub mb_w: usize,
    pub mb_h: usize,
    /// PicOrderCnt of the picture (of the field for a field picture).
    pub poc: i32,
    /// TopFieldOrderCnt / BottomFieldOrderCnt (a field picture: its own order count at its parity).
    pub field_poc: [i32; 2],
    pub structure: Structure,
    /// MBAFF frame (`mb_adaptive_frame_field_flag` and a frame picture).
    pub mbaff: bool,
    pub decoded_mbs: usize,
    pub frame: FrameRef,
    /// Decoded macroblocks per MB row.
    row_count: Vec<u16>,
    /// Leading MB rows that are fully reconstructed / deblocked / published.
    rows_recon: usize,
    rows_deblocked: usize,
    rows_published: usize,
    /// Draft mode for a non-reference picture: rows are published without deblocking.
    pub skip_deblock: bool,
}

impl PicState {
    pub fn new(frame: FrameRef, structure: Structure, mbaff: bool, poc: i32, field_poc: [i32; 2]) -> Self {
        let (mb_w, mb_h) = Self::grid(&frame, structure);
        PicState {
            planes: Planes::new(mb_w * 16, mb_h * 16),
            mbs: vec![MbState::default(); mb_w * mb_h],
            slices: Vec::new(),
            mb_w,
            mb_h,
            poc,
            field_poc,
            structure,
            mbaff,
            decoded_mbs: 0,
            frame,
            row_count: vec![0; mb_h],
            rows_recon: 0,
            rows_deblocked: 0,
            rows_published: 0,
            skip_deblock: false,
        }
    }

    fn grid(frame: &Frame, structure: Structure) -> (usize, usize) {
        if structure.is_field() && frame.interlaced { (frame.mb_w, frame.field_mb_h()) } else { (frame.mb_w, frame.mb_h()) }
    }

    /// The buffers can be reused for `structure` of `frame`.
    pub fn fits(&self, frame: &Frame, structure: Structure) -> bool {
        Self::grid(frame, structure) == (self.mb_w, self.mb_h)
    }

    /// Reuse the buffers of a finished picture of the same size for a new picture.
    pub fn reset(&mut self, frame: FrameRef, structure: Structure, mbaff: bool, poc: i32, field_poc: [i32; 2]) {
        debug_assert!(self.fits(&frame, structure));
        // start_mb() re-initialises each macroblock before use; only availability must be reset
        for m in &mut self.mbs {
            m.slice_num = u32::MAX;
        }
        self.slices.clear();
        self.poc = poc;
        self.field_poc = field_poc;
        self.structure = structure;
        self.mbaff = mbaff;
        self.decoded_mbs = 0;
        self.frame = frame;
        self.row_count.fill(0);
        self.rows_recon = 0;
        self.rows_deblocked = 0;
        self.rows_published = 0;
        self.skip_deblock = false;
    }

    /// Record a decoded macroblock (index into `mbs`) and deblock / publish rows that became final.
    #[inline]
    pub fn mb_done(&mut self, idx: usize) {
        self.decoded_mbs += 1;
        let r = idx / self.mb_w;
        let Some(c) = self.row_count.get_mut(r) else { return };
        *c += 1;
        if *c as usize >= self.mb_w && r == self.rows_recon {
            while self.rows_recon < self.mb_h && self.row_count[self.rows_recon] as usize >= self.mb_w {
                self.rows_recon += 1;
            }
            self.flush_rows(false);
        }
    }

    /// Deblock and publish rows. With `finished`, everything remaining is processed (also rows with
    /// missing macroblocks, so that waiting readers never block forever).
    pub fn flush_rows(&mut self, finished: bool) {
        // Deblocking row r must wait until row r + 1 is reconstructed (its intra prediction reads the
        // unfiltered bottom line of row r); row r is final once row r + 1 is deblocked. MBAFF
        // frames work in macroblock pair rows, deblocked in macroblock address order.
        if self.mbaff {
            let pairs_recon = self.rows_recon / 2;
            let limit = if finished { self.mb_h.div_ceil(2) } else { pairs_recon.saturating_sub(1) };
            while self.rows_deblocked / 2 < limit {
                let py = self.rows_deblocked / 2;
                if !self.skip_deblock {
                    for x in 0..self.mb_w {
                        for b in 0..2 {
                            let y = py * 2 + b;
                            if y < self.mb_h {
                                deblock::deblock_mb_mbaff(self, y * self.mb_w + x, self.mb_w);
                            }
                        }
                    }
                }
                self.rows_deblocked = (self.rows_deblocked + 2).min(self.mb_h);
            }
        } else {
            let deblock_limit = if finished { self.mb_h } else { self.rows_recon.saturating_sub(1) };
            while self.rows_deblocked < deblock_limit {
                let r = self.rows_deblocked;
                if !self.skip_deblock {
                    for addr in r * self.mb_w..(r + 1) * self.mb_w {
                        deblock::deblock_mb(self, addr, self.mb_w);
                    }
                }
                self.rows_deblocked += 1;
            }
        }
        let final_rows = if finished { self.mb_h } else { self.rows_deblocked.saturating_sub(if self.mbaff { 2 } else { 1 }) };
        if !self.frame.interlaced {
            while self.rows_published < final_rows {
                let r = self.rows_published;
                let slices = &self.slices;
                let ids =
                    |st: &MbState, l: usize, ri: i8| slices.get(st.slice_num as usize).and_then(|s| s.ref_ids[l].get(ri as usize).copied()).unwrap_or(u32::MAX);
                let row = Frame::make_row(&self.planes, r, &self.mbs, &ids);
                self.frame.publish(r, row);
                self.rows_published += 1;
            }
            return;
        }
        match self.structure.parity() {
            Some(p) => {
                if final_rows > self.rows_published {
                    self.frame.publish_field_planes(p, &self.planes, self.rows_published, final_rows);
                    self.rows_published = final_rows;
                }
            }
            None => {
                let pairs = final_rows / 2;
                if pairs > self.rows_published / 2 {
                    self.frame.publish_frame_planes(&self.planes, self.rows_published / 2, pairs);
                    self.rows_published = pairs * 2;
                }
            }
        }
        if finished {
            self.publish_col();
        }
    }

    /// Identity of the picture referenced by `ri` of list `l` of macroblock `st` (MBAFF field
    /// macroblocks index the field list of their parity).
    pub fn ref_id(&self, st: &MbState, mb_parity: u8, l: usize, ri: i8) -> u32 {
        ref_id_of(&self.slices, self.mbaff, st, mb_parity, l, ri)
    }

    /// Publish the motion data used for direct prediction by later pictures (interlaced streams).
    fn publish_col(&self) {
        let kind = match (self.structure.is_field(), self.mbaff) {
            (true, _) => ColKind::Fld,
            (false, true) => ColKind::Afrm,
            (false, false) => ColKind::Frm,
        };
        let mut col = ColPic::intra(kind, self.mb_w, self.mb_h);
        for (i, (st, c)) in self.mbs.iter().zip(col.mbs.iter_mut()).enumerate() {
            if st.slice_num == u32::MAX {
                continue;
            }
            let parity = ((i / self.mb_w) & 1) as u8;
            c.intra = st.kind.is_intra();
            c.field = st.field;
            c.mv = st.mv;
            c.ref_idx = st.ref_idx;
            for l in 0..2 {
                for b in 0..4 {
                    let ri = st.ref_idx[l][b];
                    c.ref_id[l][b] = if ri >= 0 { self.ref_id(st, parity, l, ri) } else { u32::MAX };
                }
            }
        }
        let col = Arc::new(col);
        match self.structure.parity() {
            Some(p) => self.frame.publish_col(p as usize, col),
            None => {
                self.frame.publish_col(0, col.clone());
                self.frame.publish_col(1, col);
            }
        }
    }
}

/// [`PicState::ref_id`] on the picture's slice data.
pub fn ref_id_of(slices: &[SliceInfo], mbaff: bool, st: &MbState, mb_parity: u8, l: usize, ri: i8) -> u32 {
    let Some(s) = slices.get(st.slice_num as usize) else { return u32::MAX };
    if mbaff && st.field && ri >= 0 {
        let Some(&id) = s.ref_ids[l].get(ri as usize >> 1) else { return u32::MAX };
        return (id & !3) | (1 + (mb_parity ^ (ri as u8 & 1))) as u32;
    }
    s.ref_ids[l].get(ri.max(0) as usize).copied().unwrap_or(u32::MAX)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WpMode {
    Default,
    Explicit,
    Implicit,
}

/// Prediction weight tables of a slice.
struct Weights {
    mode: WpMode,
    luma_denom: i32,
    chroma_denom: i32,
    /// [list][ref] -> (weight, offset) for luma and two chroma components.
    luma: [Vec<(i32, i32)>; 2],
    chroma: [Vec<[(i32, i32); 2]>; 2],
    /// Implicit weights (w0, w1) indexed by ref0 * n1 + ref1.
    implicit: Vec<(i32, i32)>,
    n1: usize,
    /// MBAFF: implicit weights of top / bottom field macroblocks, indexed by ref0 * 2 * n1 + ref1.
    implicit_fld: [Vec<(i32, i32)>; 2],
}

/// Current macroblock syntax (prediction part).
#[derive(Clone, Copy, Default)]
pub(crate) struct MbCur {
    pub info: MbTypeInfo,
    pub sub: [SubMbInfo; 4],
    /// rem_intra_pred_mode per block, or -1 when prev_intra_pred_mode_flag was set.
    pub rem_mode: [i8; 16],
    pub cbp: u8,
    pub ref_idx: [[i8; 4]; 2],
    /// mvd per (partition * 4 + sub-partition).
    pub mvd: [[[i16; 2]; 16]; 2],
}

/// Coefficient scratch for one macroblock.
pub(crate) struct Scratch {
    /// Luma: 16 4x4 blocks (raster block index * 16 + raster coefficient) or 4 8x8 blocks (b8 * 64 + raster).
    pub coef: [i32; 256],
    /// Chroma: [Cb, Cr][raster 2x2 block * 16 + raster coefficient].
    pub coef_c: [[i32; 64]; 2],
    /// Luma blocks with coefficients: raster 4x4 bits, or b8 bits in 8x8 mode.
    pub blk_nz: u16,
    /// Chroma blocks with coefficients (bit per raster 2x2 block).
    pub c_nz: [u8; 2],
    pub pred: [[u8; 256]; 2],
    pub pred_c: [[[u8; 64]; 2]; 2],
}

impl Scratch {
    pub fn new() -> Box<Self> {
        Box::new(Scratch { coef: [0; 256], coef_c: [[0; 64]; 2], blk_nz: 0, c_nz: [0; 2], pred: [[0; 256]; 2], pred_c: [[[0; 64]; 2]; 2] })
    }
}

/// vertMvScale of direct prediction (8.4.1.2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VScale {
    One,
    FrmToFld,
    FldToFrm,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Nb {
    pub avail: bool,
    pub ref_idx: i8,
    pub mv: [i16; 2],
}

const NB_NA: Nb = Nb { avail: false, ref_idx: -1, mv: [0, 0] };

pub struct SliceDecoder<'a> {
    pub sh: &'a SliceHeader,
    pub pps: &'a Pps,
    pub sps: &'a Sps,
    pub pic: &'a mut PicState,
    pub refs: &'a [Vec<RefPic>; 2],
    pub ls: &'a LevelScale,
    pub slice_num: u32,
    pub mb_w: usize,
    pub mb_h: usize,
    pub qp: i32,
    pub chroma_qp_offset: [i32; 2],
    // current macroblock
    /// Index of the current macroblock in `pic.mbs` (its position in the picture's MB grid).
    pub mb_addr: usize,
    /// CurrMbAddr: the macroblock address in decoding order (MBAFF: 2 * pair + bottom).
    pub cur_addr: usize,
    pub mb_x: usize,
    pub mb_y: usize,
    /// Neighbouring macroblocks A (left), B (above), C (above right), D (above left) when available.
    pub nb: [Option<usize>; 4],
    /// MBAFF: the second macroblock of a left pair whose frame / field coding differs from the
    /// current macroblock's (it holds part of the left neighbouring column).
    pub nb_left2: Option<usize>,
    pub mv_done: u16,
    pub cur: MbCur,
    pub s: Box<Scratch>,
    weights: Weights,
    /// Temporal direct: DistScaleFactor per refIdxL0, and whether pic0 is long-term: [frame list,
    /// field list of a top / bottom field macroblock (MBAFF)].
    tdirect: [Vec<(i32, bool)>; 3],
    /// CABAC: previous MB in decoding order had a non-zero mb_qp_delta.
    pub prev_qp_delta_nz: bool,
    /// Interlaced stream (`frame_mbs_only_flag` 0).
    pub interlaced: bool,
    /// MBAFF frame.
    pub mbaff: bool,
    /// The current macroblock is a field macroblock (field picture, or a field pair of an MBAFF frame).
    pub field_mb: bool,
    /// Parity of the current field macroblock (0 = top).
    pub parity: u8,
    /// Sample addressing of the current macroblock: offset of the plane view (a field of an MBAFF
    /// frame starts at line 0 or 1), its stride, and the macroblock origin inside the view.
    pub lbase: usize,
    pub lstride: usize,
    pub lx0: usize,
    pub ly0: usize,
    pub cbase: usize,
    pub cstride: usize,
    pub cx0: usize,
    pub cy0: usize,
    /// Scans of the current macroblock (frame or field).
    pub scan4: &'static [u8; 16],
    pub scan8: &'static [u8; 64],
    /// Motion data of the co-located picture (interlaced B slices): [top, bottom] parity.
    col: [Option<Arc<ColPic>>; 2],
}

/// Interpolate a bw x bh luma block at (px, py) with motion vector `mv` from reference `f` into `y`,
/// and the corresponding chroma blocks into `c` (stride `cs`). `cdy` is added to the vertical
/// chroma vector (field prediction from the opposite parity, Table 8-9).
#[allow(clippy::too_many_arguments)]
#[inline]
fn predict_block(f: &RefPic, mv: [i16; 2], px: usize, py: usize, bw: usize, bh: usize, y: (&mut [u8], usize), c: [&mut [u8]; 2], cs: usize, cdy: i32) {
    const LW: usize = 24;
    let mut win = [0u8; LW * 21];
    let ix = px as i32 + (mv[0] as i32 >> 2);
    let iy = py as i32 + (mv[1] as i32 >> 2);
    f.luma_window(ix - 2, iy - 2, bw + 5, bh + 5, &mut win, LW);
    inter::mc_luma_win(&win, LW, (mv[0] & 3) as u32, (mv[1] & 3) as u32, bw, bh, y.0, y.1);
    let cmy = mv[1] as i32 + cdy;
    let cx = (px / 2) as i32 + (mv[0] as i32 >> 3);
    let cy = (py / 2) as i32 + (cmy >> 3);
    let (cw, ch) = (bw / 2, bh / 2);
    let (fx, fy) = ((mv[0] & 7) as u32, (cmy & 7) as u32);
    for (comp, out) in c.into_iter().enumerate() {
        let mut w = [0u8; 16 * 9];
        f.chroma_window(comp, cx, cy, cw + 1, ch + 1, &mut w, 16);
        inter::mc_chroma_win(&w, 16, fx, fy, cw, ch, out, cs);
    }
}

#[inline(always)]
fn median(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(a.min(b).max(c))
}

/// DistScaleFactor-style weights (8.4.1.2.3 / 8.4.2.3.1) of `cur` between `poc0` and `poc1`.
fn dist_scale(cur: i32, poc0: i32, poc1: i32) -> Option<i32> {
    let tb = (cur - poc0).clamp(-128, 127);
    let td = (poc1 - poc0).clamp(-128, 127);
    if td == 0 {
        return None;
    }
    let tx = (16384 + (td / 2).abs()) / td;
    Some(((tb * tx + 32) >> 6).clamp(-1024, 1023))
}

impl<'a> SliceDecoder<'a> {
    pub fn new(sh: &'a SliceHeader, pps: &'a Pps, sps: &'a Sps, pic: &'a mut PicState, refs: &'a [Vec<RefPic>; 2], ls: &'a LevelScale) -> Result<Self> {
        let slice_num = pic.slices.len() as u32;
        let chroma_qp_offset = [pps.chroma_qp_index_offset, pps.second_chroma_qp_index_offset];
        pic.slices.push(SliceInfo {
            disable_deblocking_filter_idc: sh.disable_deblocking_filter_idc,
            alpha_offset: sh.slice_alpha_c0_offset_div2 * 2,
            beta_offset: sh.slice_beta_offset_div2 * 2,
            ref_ids: [refs[0].iter().map(|r| r.id()).collect(), refs[1].iter().map(|r| r.id()).collect()],
        });
        let mbaff = pic.mbaff;
        let weights = Self::build_weights(sh, pps, refs, pic.poc, mbaff.then_some(pic.field_poc));
        let mut tdirect = [Vec::new(), Vec::new(), Vec::new()];
        let mut col = [None, None];
        if sh.slice_type == SliceType::B && !refs[1].is_empty() {
            if !sh.direct_spatial_mv_pred {
                let t = |cur: i32, l0: &[RefPic], r1: &RefPic| -> Vec<(i32, bool)> {
                    l0.iter()
                        .map(|r0| if r0.long_term { (256, true) } else { dist_scale(cur, r0.poc(), r1.poc()).map_or((256, true), |d| (d, false)) })
                        .collect()
                };
                tdirect[0] = t(pic.poc, &refs[0], &refs[1][0]);
                if mbaff {
                    for p in 0..2u8 {
                        let l0: Vec<RefPic> = refs[0].iter().flat_map(|r| [r.field_of(p), r.field_of(p ^ 1)]).collect();
                        tdirect[1 + p as usize] = t(pic.field_poc[p as usize], &l0, &refs[1][0].field_of(p));
                    }
                }
            }
            if pic.frame.interlaced {
                let f = &refs[1][0].frame;
                col = [f.col(0).cloned(), f.col(1).cloned()];
            }
        }
        let (mb_w, mb_h) = (pic.mb_w, pic.mb_h);
        let interlaced = pic.frame.interlaced;
        let field_pic = pic.structure.is_field();
        let parity = pic.structure.parity().unwrap_or(0);
        let (scan4, scan8) = if field_pic { (&FIELD4, &FIELD8) } else { (&ZIGZAG4, &ZIGZAG8) };
        let (pw, cw) = (pic.planes.width, pic.planes.cwidth);
        Ok(SliceDecoder {
            sh,
            pps,
            sps,
            pic,
            refs,
            ls,
            slice_num,
            mb_w,
            mb_h,
            qp: sh.qp(pps),
            chroma_qp_offset,
            mb_addr: 0,
            cur_addr: 0,
            mb_x: 0,
            mb_y: 0,
            nb: [None; 4],
            nb_left2: None,
            mv_done: 0,
            cur: MbCur::default(),
            s: Scratch::new(),
            weights,
            tdirect,
            prev_qp_delta_nz: false,
            interlaced,
            mbaff,
            field_mb: field_pic,
            parity,
            lbase: 0,
            lstride: pw,
            lx0: 0,
            ly0: 0,
            cbase: 0,
            cstride: cw,
            cx0: 0,
            cy0: 0,
            scan4,
            scan8,
            col,
        })
    }

    /// `mbaff_field_poc`: for MBAFF frames, the field order counts of the current frame (implicit
    /// weights of field macroblocks use field distances).
    fn build_weights(sh: &SliceHeader, pps: &Pps, refs: &[Vec<RefPic>; 2], cur_poc: i32, mbaff_field_poc: Option<[i32; 2]>) -> Weights {
        let mut w = Weights {
            mode: WpMode::Default,
            luma_denom: 0,
            chroma_denom: 0,
            luma: [Vec::new(), Vec::new()],
            chroma: [Vec::new(), Vec::new()],
            implicit: Vec::new(),
            n1: refs[1].len(),
            implicit_fld: [Vec::new(), Vec::new()],
        };
        let explicit = (pps.weighted_pred && sh.slice_type.is_p()) || (pps.weighted_bipred_idc == 1 && sh.slice_type.is_b());
        if explicit {
            if let Some(t) = &sh.pred_weight_table {
                w.mode = WpMode::Explicit;
                w.luma_denom = t.luma_log2_denom as i32;
                w.chroma_denom = t.chroma_log2_denom as i32;
                for (l, entries) in [&t.l0, &t.l1].into_iter().enumerate() {
                    w.luma[l] = entries.iter().map(|e| (e.luma_weight, e.luma_offset)).collect();
                    w.chroma[l] = entries.iter().map(|e| [(e.chroma_weight[0], e.chroma_offset[0]), (e.chroma_weight[1], e.chroma_offset[1])]).collect();
                }
            }
        } else if pps.weighted_bipred_idc == 2 && sh.slice_type.is_b() {
            w.mode = WpMode::Implicit;
            let table = |cur: i32, l0: &[RefPic], l1: &[RefPic]| -> Vec<(i32, i32)> {
                let mut out = Vec::with_capacity(l0.len() * l1.len());
                for r0 in l0 {
                    for r1 in l1 {
                        let pair = match dist_scale(cur, r0.poc(), r1.poc()) {
                            Some(dsf) if !r0.long_term && !r1.long_term && (-64..=128).contains(&(dsf >> 2)) => (64 - (dsf >> 2), dsf >> 2),
                            _ => (32, 32),
                        };
                        out.push(pair);
                    }
                }
                out
            };
            w.implicit = table(cur_poc, &refs[0], &refs[1]);
            if let Some(fp) = mbaff_field_poc {
                for p in 0..2u8 {
                    let fl = |l: &[RefPic]| -> Vec<RefPic> { l.iter().flat_map(|r| [r.field_of(p), r.field_of(p ^ 1)]).collect() };
                    w.implicit_fld[p as usize] = table(fp[p as usize], &fl(&refs[0]), &fl(&refs[1]));
                }
            }
        }
        w
    }

    // ------------------------------------------------------------------------------------------
    // Neighbour helpers
    // ------------------------------------------------------------------------------------------

    /// Set up the current macroblock address and neighbour availability (non-MBAFF pictures).
    pub fn start_mb(&mut self, addr: usize) {
        self.mb_addr = addr;
        self.cur_addr = addr;
        self.mb_x = addr % self.mb_w;
        self.mb_y = addr / self.mb_w;
        let sn = self.slice_num;
        let mbs = &self.pic.mbs;
        let w = self.mb_w;
        let (x, y) = (self.mb_x, self.mb_y);
        let ok = |a: usize| if mbs[a].slice_num == sn { Some(a) } else { None };
        self.nb = [
            if x > 0 { ok(addr - 1) } else { None },
            if y > 0 { ok(addr - w) } else { None },
            if y > 0 && x + 1 < w { ok(addr - w + 1) } else { None },
            if y > 0 && x > 0 { ok(addr - w - 1) } else { None },
        ];
        self.mv_done = 0;
        self.lx0 = x * 16;
        self.ly0 = y * 16;
        self.cx0 = x * 8;
        self.cy0 = y * 8;
        let field = self.field_mb;
        self.pic.mbs[addr] = MbState { slice_num: sn, field, ..MbState::default() };
    }

    /// Reference picture `ri` of list `l` for the current macroblock: MBAFF field macroblocks use
    /// field lists (8.4.2.1): field `parity ^ (ri & 1)` of frame `ri / 2`.
    #[inline]
    pub fn ref_pic(&self, l: usize, ri: i8) -> Option<std::borrow::Cow<'a, RefPic>> {
        if ri < 0 {
            return None;
        }
        let refs: &'a [Vec<RefPic>; 2] = self.refs;
        if self.mbaff && self.field_mb {
            let r = refs[l].get(ri as usize >> 1)?;
            return Some(std::borrow::Cow::Owned(r.field_of(self.parity ^ (ri as u8 & 1))));
        }
        refs[l].get(ri as usize).map(std::borrow::Cow::Borrowed)
    }

    /// Number of entries of list `l` for the current macroblock.
    #[inline]
    pub fn num_refs(&self, l: usize) -> usize {
        if self.mbaff && self.field_mb { 2 * self.refs[l].len() } else { self.refs[l].len() }
    }

    #[inline]
    pub fn mb(&self) -> &MbState {
        &self.pic.mbs[self.mb_addr]
    }
    #[inline]
    pub fn mb_mut(&mut self) -> &mut MbState {
        &mut self.pic.mbs[self.mb_addr]
    }

    /// Neighbour MB usable for intra prediction (constrained_intra_pred aware).
    fn intra_nb(&self, n: usize) -> bool {
        let ok = |m: usize| !self.pps.constrained_intra_pred || self.pic.mbs[m].kind.is_intra();
        match self.nb[n] {
            None => false,
            Some(a) => ok(a) && (n != 0 || self.nb_left2.is_none_or(ok)),
        }
    }

    fn chroma_qp(&self, qp: i32, c: usize) -> i32 {
        QPC_TABLE[(qp + self.chroma_qp_offset[c]).clamp(0, 51) as usize] as i32
    }

    /// nC for a luma 4x4 block (raster index) - 9.2.1.
    fn nc_luma(&self, raster: usize) -> i32 {
        let (bx, by) = (raster & 3, raster >> 2);
        let cur = self.mb();
        let (a, b) = if self.mbaff {
            let nnz = |n: Option<(usize, usize)>| n.map(|(m, r)| self.pic.mbs[m].nnz[r]);
            (
                if bx > 0 { Some(cur.nnz[raster - 1]) } else { nnz(self.nb_blk4_mbaff(-1, by as i32)) },
                if by > 0 { Some(cur.nnz[raster - 4]) } else { nnz(self.nb_blk4_mbaff(bx as i32, -1)) },
            )
        } else {
            (
                if bx > 0 { Some(cur.nnz[raster - 1]) } else { self.nb[0].map(|m| self.pic.mbs[m].nnz[by * 4 + 3]) },
                if by > 0 { Some(cur.nnz[raster - 4]) } else { self.nb[1].map(|m| self.pic.mbs[m].nnz[12 + bx]) },
            )
        };
        match (a, b) {
            (Some(a), Some(b)) => (a as i32 + b as i32 + 1) >> 1,
            (Some(a), None) => a as i32,
            (None, Some(b)) => b as i32,
            _ => 0,
        }
    }

    fn nc_chroma(&self, c: usize, raster: usize) -> i32 {
        let (bx, by) = (raster & 1, raster >> 1);
        let cur = self.mb();
        let (a, b) = if self.mbaff {
            let nnz = |n: Option<(usize, usize)>| n.map(|(m, r)| self.pic.mbs[m].nnz_c[c][r]);
            (
                if bx > 0 { Some(cur.nnz_c[c][raster - 1]) } else { nnz(self.nb_chroma_blk_mbaff(-1, by as i32)) },
                if by > 0 { Some(cur.nnz_c[c][raster - 2]) } else { nnz(self.nb_chroma_blk_mbaff(bx as i32, -1)) },
            )
        } else {
            (
                if bx > 0 { Some(cur.nnz_c[c][raster - 1]) } else { self.nb[0].map(|m| self.pic.mbs[m].nnz_c[c][by * 2 + 1]) },
                if by > 0 { Some(cur.nnz_c[c][raster - 2]) } else { self.nb[1].map(|m| self.pic.mbs[m].nnz_c[c][2 + bx]) },
            )
        };
        match (a, b) {
            (Some(a), Some(b)) => (a as i32 + b as i32 + 1) >> 1,
            (Some(a), None) => a as i32,
            (None, Some(b)) => b as i32,
            _ => 0,
        }
    }

    /// Motion data of the neighbouring 4x4 block at (x, y) (4x4 units, relative to the current MB).
    pub(crate) fn nb_motion(&self, list: usize, x: i32, y: i32) -> Nb {
        self.nb_motion_luma(list, if x < 0 { -1 } else { x * 4 }, if y < 0 { -1 } else { y * 4 })
    }

    /// Motion data of the neighbouring block covering luma location (xl, yl) relative to the
    /// current MB (MBAFF frames map locations, not 4x4 blocks: 6.4.12.2).
    pub(crate) fn nb_motion_luma(&self, list: usize, xl: i32, yl: i32) -> Nb {
        let (x, y) = (xl >> 2, yl >> 2);
        if self.mbaff && (xl < 0 || yl < 0 || xl > 15) {
            if (x > 3 && y >= 0) || (x < 0 && y > 3) {
                return NB_NA;
            }
            let Some((m, xw, yw)) = self.nb_loc_mbaff(xl, yl, 16, 16) else { return NB_NA };
            let r = ((yw >> 2) * 4 + (xw >> 2)) as usize;
            let st = &self.pic.mbs[m];
            let b8 = (r >> 3) * 2 + ((r & 3) >> 1);
            let ri = st.ref_idx[list][b8];
            if ri < 0 {
                return Nb { avail: true, ref_idx: -1, mv: [0, 0] };
            }
            let (ri, mv) = self.mbaff_scale(st, ri, st.mv[list][r]);
            return Nb { avail: true, ref_idx: ri, mv };
        }
        let (mb, bx, by) = if x < 0 {
            if y < 0 {
                (self.nb[3], 3, 3)
            } else if y > 3 {
                (None, 0, 0)
            } else {
                (self.nb[0], 3, y as usize)
            }
        } else if x > 3 {
            if y < 0 { (self.nb[2], 0, 3) } else { (None, 0, 0) }
        } else if y < 0 {
            (self.nb[1], x as usize, 3)
        } else {
            let r = (y * 4 + x) as usize;
            if self.mv_done & (1 << r) == 0 {
                return NB_NA;
            }
            (Some(self.mb_addr), x as usize, y as usize)
        };
        let Some(m) = mb else { return NB_NA };
        let st = &self.pic.mbs[m];
        let b8 = (by >> 1) * 2 + (bx >> 1);
        let r = st.ref_idx[list][b8];
        if r < 0 {
            return Nb { avail: true, ref_idx: -1, mv: [0, 0] };
        }
        Nb { avail: true, ref_idx: r, mv: st.mv[list][by * 4 + bx] }
    }

    /// Returns (A, B, C) neighbour motion for a partition at (x, y) of width w (4x4 units), with C
    /// replaced by D when unavailable.
    fn nb_abc(&self, list: usize, x: usize, y: usize, w: usize) -> (Nb, Nb, Nb) {
        // luma locations of A, B, C, D (6.4.11.7)
        let (x, y, w) = (x as i32 * 4, y as i32 * 4, w as i32 * 4);
        let a = self.nb_motion_luma(list, x - 1, y);
        let b = self.nb_motion_luma(list, x, y - 1);
        let mut c = self.nb_motion_luma(list, x + w, y - 1);
        if !c.avail {
            c = self.nb_motion_luma(list, x - 1, y - 1);
        }
        (a, b, c)
    }

    /// Luma motion vector prediction (8.4.1.3). `shape`: 1 = 16x8, 2 = 8x16, 0 = other.
    #[allow(clippy::too_many_arguments)]
    fn mvp(&self, list: usize, x: usize, y: usize, w: usize, ref_idx: i8, shape: u8, part: usize) -> [i16; 2] {
        let (a, b, c) = self.nb_abc(list, x, y, w);
        if shape == 1 {
            if part == 0 && b.ref_idx == ref_idx {
                return b.mv;
            }
            if part == 1 && a.ref_idx == ref_idx {
                return a.mv;
            }
        } else if shape == 2 {
            if part == 0 && a.ref_idx == ref_idx {
                return a.mv;
            }
            if part == 1 && c.ref_idx == ref_idx {
                return c.mv;
            }
        }
        Self::median_pred(a, b, c, ref_idx)
    }

    fn median_pred(a: Nb, mut b: Nb, mut c: Nb, ref_idx: i8) -> [i16; 2] {
        if !b.avail && !c.avail && a.avail {
            b = a;
            c = a;
        }
        let ma = a.ref_idx == ref_idx;
        let mb = b.ref_idx == ref_idx;
        let mc = c.ref_idx == ref_idx;
        match (ma, mb, mc) {
            (true, false, false) => a.mv,
            (false, true, false) => b.mv,
            (false, false, true) => c.mv,
            _ => [median(a.mv[0], b.mv[0], c.mv[0]), median(a.mv[1], b.mv[1], c.mv[1])],
        }
    }

    /// Store motion for a rectangle (4x4 units) of the current MB and mark it decoded.
    fn set_motion(&mut self, x: usize, y: usize, w: usize, h: usize, refs: [i8; 2], mvs: [[i16; 2]; 2]) {
        let st = &mut self.pic.mbs[self.mb_addr];
        for yy in y..y + h {
            for xx in x..x + w {
                let r = yy * 4 + xx;
                for l in 0..2 {
                    st.mv[l][r] = if refs[l] >= 0 { mvs[l] } else { [0, 0] };
                }
                self.mv_done |= 1 << r;
            }
        }
        for l in 0..2 {
            // ref_idx is per 8x8; rectangles never straddle 8x8 blocks with different refs
            for yy in (y..y + h).step_by(1) {
                for xx in x..x + w {
                    st.ref_idx[l][(yy >> 1) * 2 + (xx >> 1)] = refs[l];
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // Direct prediction (8.4.1.2)
    // ------------------------------------------------------------------------------------------

    /// Co-located (mvCol, refIdxCol, referenced picture id, vertMvScale) for 4x4 block `raster` of
    /// the current MB (8.4.1.2.1).
    fn colocated(&self, raster: usize) -> ([i16; 2], i8, u32, VScale) {
        const NONE: ([i16; 2], i8, u32, VScale) = ([0, 0], -1, u32::MAX, VScale::One);
        let Some(col) = self.refs[1].first() else { return NONE };
        if self.interlaced {
            return self.colocated_interlaced(col, raster);
        }
        let raster = if self.sps.direct_8x8_inference {
            let (bx, by) = (raster & 3, raster >> 2);
            (if by >= 2 { 12 } else { 0 }) + if bx >= 2 { 3 } else { 0 }
        } else {
            raster
        };
        if col.frame.mb_w != self.mb_w || self.mb_y >= col.frame.mb_h() || col.frame.interlaced {
            return NONE;
        }
        let m = col.frame.row(self.mb_y);
        let mb = self.mb_x;
        if m.intra[mb] {
            return NONE;
        }
        let b8 = mb * 4 + (raster >> 3) * 2 + ((raster & 3) >> 1);
        let l = if m.ref_idx[0][b8] >= 0 { 0 } else { 1 };
        (m.mv[l][mb * 16 + raster], m.ref_idx[l][b8], m.ref_id[l][b8], VScale::One)
    }

    /// [`Self::colocated`] for interlaced streams: Tables 8-6 and 8-8.
    fn colocated_interlaced(&self, r1: &RefPic, raster: usize) -> ([i16; 2], i8, u32, VScale) {
        const NONE: ([i16; 2], i8, u32, VScale) = ([0, 0], -1, u32::MAX, VScale::One);
        let (bx, by) = (raster & 3, raster >> 2);
        let (x_col, y_col) = if self.sps.direct_8x8_inference { (if bx >= 2 { 12 } else { 0 }, if by >= 2 { 12 } else { 0 }) } else { (bx * 4, by * 4) };
        // which field of RefPicList1[0] (frame decoding of a complementary field pair)
        let top_abs = (r1.field_poc[0] - self.pic.poc).abs();
        let bottom_abs = (r1.field_poc[1] - self.pic.poc).abs();
        let poc_parity = if top_abs < bottom_abs { 0 } else { 1 };
        let field_pic = self.pic.structure.is_field();
        let colpic = if field_pic {
            self.col[r1.structure.parity().unwrap_or(0) as usize & 1].as_ref()
        } else {
            match self.col[0].as_ref() {
                Some(c) if c.kind == ColKind::Fld => {
                    let p = if self.field_mb { self.parity } else { poc_parity };
                    self.col[p as usize].as_ref()
                }
                c => c,
            }
        };
        let Some(colpic) = colpic else { return NONE };
        let (mx, my) = (self.mb_x, self.mb_y);
        let frm_to_fld = |y: usize| (y, (2 * y_col) % 16, VScale::FrmToFld);
        let (cy, ym, scale) = if field_pic {
            match colpic.kind {
                ColKind::Fld => (my, y_col, VScale::One),
                ColKind::Frm => frm_to_fld(2 * my + y_col / 8),
                ColKind::Afrm => {
                    if colpic.mb(mx, 2 * my).field {
                        (2 * my + self.parity as usize, y_col, VScale::One)
                    } else {
                        frm_to_fld(2 * my + y_col / 8)
                    }
                }
            }
        } else if !self.mbaff {
            match colpic.kind {
                ColKind::Fld => (my / 2, 8 * (my % 2) + 4 * (y_col / 8), VScale::FldToFrm),
                _ => (my, y_col, VScale::One),
            }
        } else {
            let (py, bottom) = (my / 2, my % 2);
            match colpic.kind {
                ColKind::Fld => {
                    if self.field_mb {
                        (py, y_col, VScale::One)
                    } else {
                        (py, 8 * bottom + 4 * (y_col / 8), VScale::FldToFrm)
                    }
                }
                _ => {
                    let x_field = colpic.mb(mx, my).field;
                    match (self.field_mb, x_field) {
                        (false, false) | (true, true) => (my, y_col, VScale::One),
                        (false, true) => (2 * py + poc_parity as usize, 8 * bottom + 4 * (y_col / 8), VScale::FldToFrm),
                        (true, false) => frm_to_fld(2 * py + y_col / 8),
                    }
                }
            }
        };
        if colpic.mb_w != self.mb_w || cy >= colpic.mb_h {
            return NONE;
        }
        let c = colpic.mb(mx, cy);
        if c.intra {
            return NONE;
        }
        let r = (ym / 4) * 4 + x_col / 4;
        let b8 = (ym / 8) * 2 + x_col / 8;
        let l = if c.ref_idx[0][b8] >= 0 { 0 } else { 1 };
        (c.mv[l][r], c.ref_idx[l][b8], c.ref_id[l][b8], scale)
    }

    /// Spatial direct: MB-level reference indices, predicted mvs and directZeroPrediction (8.4.1.2.2).
    fn spatial_direct_base(&self) -> ([i8; 2], [[i16; 2]; 2], bool) {
        let mut refs = [-1i8; 2];
        let mut nbs = [(NB_NA, NB_NA, NB_NA); 2];
        let min_pos = |x: i8, y: i8| if x >= 0 && y >= 0 { x.min(y) } else { x.max(y) };
        for (l, nb) in nbs.iter_mut().enumerate() {
            *nb = self.nb_abc(l, 0, 0, 4);
            refs[l] = min_pos(nb.0.ref_idx, min_pos(nb.1.ref_idx, nb.2.ref_idx));
        }
        let mut mvs = [[0i16; 2]; 2];
        if refs[0] < 0 && refs[1] < 0 {
            return ([0, 0], mvs, true);
        }
        for l in 0..2 {
            if refs[l] >= 0 {
                let (a, b, c) = nbs[l];
                mvs[l] = Self::median_pred(a, b, c, refs[l]);
            }
        }
        (refs, mvs, false)
    }

    /// Derive direct-mode motion for the 8x8 blocks in `b8_mask` of the current MB.
    fn derive_direct(&mut self, b8_mask: u8) -> Result<()> {
        let spatial = self.sh.direct_spatial_mv_pred;
        let (base_refs, base_mvs, direct_zero) = if spatial { self.spatial_direct_base() } else { ([0, 0], [[0, 0]; 2], false) };
        let col_short_term = self.refs[1].first().map(|r| !r.long_term).unwrap_or(false);
        for b8 in 0..4 {
            if b8_mask & (1 << b8) == 0 {
                continue;
            }
            let (x8, y8) = ((b8 & 1) * 2, (b8 >> 1) * 2);
            let unit = if self.sps.direct_8x8_inference { 2 } else { 1 };
            for sub in (0..4).filter(|s| unit == 1 || *s == 0) {
                let (x, y) = (x8 + (sub & 1), y8 + (sub >> 1));
                let raster = y * 4 + x;
                let (mv_col, ref_col, id_col, scale) = self.colocated(raster);
                let (refs, mvs) = if spatial {
                    let col_zero = col_short_term && ref_col == 0 && (-1..=1).contains(&mv_col[0]) && (-1..=1).contains(&mv_col[1]);
                    let mut mvs = base_mvs;
                    for l in 0..2 {
                        if direct_zero || base_refs[l] < 0 || (base_refs[l] == 0 && col_zero) {
                            mvs[l] = [0, 0];
                        }
                    }
                    (base_refs, mvs)
                } else {
                    // 8.4.1.2.3: the picture refIdxCol referred to, as seen from the current
                    // picture / macroblock, and the vertical vector scaled to its structure
                    let (target, mv_col) = match scale {
                        VScale::One => (id_col, mv_col),
                        VScale::FrmToFld => ((id_col & !3) | (1 + self.parity as u32), [mv_col[0], mv_col[1] / 2]),
                        VScale::FldToFrm => (id_col & !3, [mv_col[0], mv_col[1].wrapping_mul(2)]),
                    };
                    let ref0 = if ref_col < 0 {
                        0
                    } else {
                        (0..self.num_refs(0)).find(|&i| self.ref_pic(0, i as i8).is_some_and(|r| r.id() == target)).unwrap_or(0) as i8
                    };
                    let t = if self.mbaff && self.field_mb { 1 + self.parity as usize } else { 0 };
                    let (dsf, ltr) = self.tdirect[t].get(ref0 as usize).copied().unwrap_or((256, true));
                    let (mv0, mv1) = if ltr {
                        (mv_col, [0, 0])
                    } else {
                        let s = |c: i16| ((dsf * c as i32 + 128) >> 8) as i16;
                        let m0 = [s(mv_col[0]), s(mv_col[1])];
                        (m0, [m0[0] - mv_col[0], m0[1] - mv_col[1]])
                    };
                    ([ref0, 0], [mv0, mv1])
                };
                self.set_motion(x, y, unit, unit, refs, mvs);
            }
        }
        let st = &mut self.pic.mbs[self.mb_addr];
        st.direct8x8 |= b8_mask;
        Ok(())
    }

    // ------------------------------------------------------------------------------------------
    // Motion vector derivation for a parsed inter macroblock
    // ------------------------------------------------------------------------------------------

    /// P_Skip motion (8.4.1.1).
    pub fn derive_p_skip(&mut self) {
        let a = self.nb_motion(0, -1, 0);
        let b = self.nb_motion(0, 0, -1);
        let mv = if self.nb[0].is_none() || self.nb[1].is_none() || (a.ref_idx == 0 && a.mv == [0, 0]) || (b.ref_idx == 0 && b.mv == [0, 0]) {
            [0, 0]
        } else {
            self.mvp(0, 0, 0, 4, 0, 0, 0)
        };
        self.set_motion(0, 0, 4, 4, [0, -1], [mv, [0, 0]]);
    }

    /// Derive mvs of the current (non-skip) inter MB from `self.cur` (ref_idx, mvd).
    pub fn derive_inter_motion(&mut self) -> Result<()> {
        let info = self.cur.info;
        if info.kind == MbKind::BDirect16x16 || info.kind == MbKind::BSkip {
            return self.derive_direct(0xf);
        }
        let n = [self.num_refs(0), self.num_refs(1)];
        if info.part != Part::P8x8 {
            let shape = match info.part {
                Part::P16x8 => 1,
                Part::P8x16 => 2,
                _ => 0,
            };
            for p in 0..info.part.num_parts() {
                let (x, y, w, h) = info.part.rect(p);
                let mut refs = [-1i8; 2];
                let mut mvs = [[0i16; 2]; 2];
                for l in 0..2 {
                    if info.pred[p] & (1 << l) != 0 {
                        let r = self.cur.ref_idx[l][p];
                        ensure!((r as usize) < n[l], "ref_idx {r} out of range (list {l}, size {})", n[l]);
                        let mvp = self.mvp(l, x, y, w, r, shape, p);
                        let d = self.cur.mvd[l][p * 4];
                        refs[l] = r;
                        mvs[l] = [mvp[0].wrapping_add(d[0]), mvp[1].wrapping_add(d[1])];
                    }
                }
                self.set_motion(x, y, w, h, refs, mvs);
            }
            return Ok(());
        }
        for p in 0..4 {
            let sub = self.cur.sub[p];
            let (x8, y8) = ((p & 1) * 2, (p >> 1) * 2);
            if sub.direct {
                self.derive_direct(1 << p)?;
                continue;
            }
            let mut refs = [-1i8; 2];
            for (l, r) in refs.iter_mut().enumerate() {
                if sub.pred & (1 << l) != 0 {
                    let v = self.cur.ref_idx[l][p];
                    ensure!((v as usize) < n[l], "ref_idx {v} out of range (list {l}, size {})", n[l]);
                    *r = v;
                }
            }
            for j in 0..sub.shape.num_parts() {
                let (sx, sy, w, h) = sub.shape.rect(j);
                let (x, y) = (x8 + sx, y8 + sy);
                let mut mvs = [[0i16; 2]; 2];
                for l in 0..2 {
                    if refs[l] >= 0 {
                        let mvp = self.mvp(l, x, y, w, refs[l], 0, 0);
                        let d = self.cur.mvd[l][p * 4 + j];
                        mvs[l] = [mvp[0].wrapping_add(d[0]), mvp[1].wrapping_add(d[1])];
                    }
                }
                self.set_motion(x, y, w, h, refs, mvs);
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------------------------------
    // Inter prediction
    // ------------------------------------------------------------------------------------------

    fn weight_for(&self, refs: [i8; 2], comp: usize) -> Weight {
        let w = &self.weights;
        match w.mode {
            WpMode::Default => Weight::Default,
            WpMode::Implicit => {
                if refs[0] >= 0 && refs[1] >= 0 {
                    let (table, n1) = if self.mbaff && self.field_mb { (&w.implicit_fld[self.parity as usize & 1], 2 * w.n1) } else { (&w.implicit, w.n1) };
                    let (w0, w1) = table.get(refs[0] as usize * n1 + refs[1] as usize).copied().unwrap_or((32, 32));
                    Weight::Weighted { log_wd: 5, w0, w1, o0: 0, o1: 0 }
                } else {
                    Weight::Default
                }
            }
            WpMode::Explicit => {
                let shift = (self.mbaff && self.field_mb) as usize;
                let get = |l: usize| -> (i32, i32) {
                    if refs[l] < 0 {
                        return (0, 0);
                    }
                    let r = refs[l] as usize >> shift;
                    if comp == 0 {
                        w.luma[l].get(r).copied().unwrap_or((1 << w.luma_denom, 0))
                    } else {
                        w.chroma[l].get(r).map(|c| c[comp - 1]).unwrap_or((1 << w.chroma_denom, 0))
                    }
                };
                let (w0, o0) = get(0);
                let (w1, o1) = get(1);
                let log_wd = if comp == 0 { w.luma_denom } else { w.chroma_denom };
                Weight::Weighted { log_wd, w0, w1, o0, o1 }
            }
        }
    }

    /// Vertical chroma vector offset for prediction from field `rp` (Table 8-9).
    #[inline]
    fn chroma_dy(&self, rp: &RefPic) -> i32 {
        match rp.structure.parity() {
            Some(p) if self.field_mb => 2 * (self.parity as i32 - p as i32),
            _ => 0,
        }
    }

    /// Motion-compensated prediction of a rectangle (4x4 units) written into the picture.
    fn mc_rect(&mut self, x4: usize, y4: usize, w4: usize, h4: usize) {
        let st = &self.pic.mbs[self.mb_addr];
        let r = y4 * 4 + x4;
        let b8 = (y4 >> 1) * 2 + (x4 >> 1);
        let refs = [st.ref_idx[0][b8], st.ref_idx[1][b8]];
        let mvs = [st.mv[0][r], st.mv[1][r]];
        let (bw, bh) = (w4 * 4, h4 * 4);
        let wts = [self.weight_for(refs, 0), self.weight_for(refs, 1), self.weight_for(refs, 2)];
        let px = self.lx0 + x4 * 4;
        let py = self.ly0 + y4 * 4;
        let rp = [self.ref_pic(0, refs[0]), self.ref_pic(1, refs[1])];
        let use0 = rp[0].is_some();
        let use1 = rp[1].is_some();
        let ys = self.lstride;
        let cs = self.cstride;
        let (cx, cy) = (self.cx0 + x4 * 2, self.cy0 + y4 * 2);
        let yo = self.lbase + py * ys + px;
        let co = self.cbase + cy * cs + cx;
        if use0 != use1 && wts.iter().all(|w| *w == Weight::Default) {
            // single-list unweighted prediction: interpolate straight into the picture
            let l = if use0 { 0 } else { 1 };
            let Some(f) = rp[l].as_deref() else { return };
            let cdy = self.chroma_dy(f);
            let planes = &mut self.pic.planes;
            predict_block(f, mvs[l], px, py, bw, bh, (&mut planes.y[yo..], ys), [&mut planes.cb[co..], &mut planes.cr[co..]], cs, cdy);
            return;
        }
        let cdy = [rp[0].as_deref().map_or(0, |f| self.chroma_dy(f)), rp[1].as_deref().map_or(0, |f| self.chroma_dy(f))];
        let s = &mut *self.s;
        for l in 0..2 {
            let Some(f) = rp[l].as_deref() else { continue };
            let [pc0, pc1] = &mut s.pred_c[l];
            predict_block(f, mvs[l], px, py, bw, bh, (&mut s.pred[l][..], 16), [&mut pc0[..], &mut pc1[..]], 8, cdy[l]);
        }
        inter::weighted_store(
            &mut self.pic.planes.y[yo..],
            ys,
            if use0 { Some(&s.pred[0][..]) } else { None },
            if use1 { Some(&s.pred[1][..]) } else { None },
            16,
            bw,
            bh,
            wts[0],
        );
        for c in 0..2 {
            let wc = wts[c + 1];
            let a = if use0 { Some(&s.pred_c[0][c][..]) } else { None };
            let b = if use1 { Some(&s.pred_c[1][c][..]) } else { None };
            let plane = if c == 0 { &mut self.pic.planes.cb } else { &mut self.pic.planes.cr };
            inter::weighted_store(&mut plane[co..], cs, a, b, 8, bw / 2, bh / 2, wc);
        }
    }

    /// All 16 blocks of the current MB share references and motion vectors.
    fn uniform_motion(&self) -> bool {
        let st = self.mb();
        (0..2).all(|l| {
            let r = st.ref_idx[l];
            r[1] == r[0] && r[2] == r[0] && r[3] == r[0] && st.mv[l].iter().all(|m| *m == st.mv[l][0])
        })
    }

    /// Inter prediction for the whole current MB, using the stored motion.
    pub fn predict_inter(&mut self) {
        let info = self.cur.info;
        let direct_unit = if self.sps.direct_8x8_inference { 2 } else { 1 };
        match info.kind {
            MbKind::PSkip => self.mc_rect(0, 0, 4, 4),
            MbKind::BSkip | MbKind::BDirect16x16 if self.uniform_motion() => self.mc_rect(0, 0, 4, 4),
            MbKind::BSkip | MbKind::BDirect16x16 => {
                for y in (0..4).step_by(direct_unit) {
                    for x in (0..4).step_by(direct_unit) {
                        self.mc_rect(x, y, direct_unit, direct_unit);
                    }
                }
            }
            _ => {
                if info.part != Part::P8x8 {
                    for p in 0..info.part.num_parts() {
                        let (x, y, w, h) = info.part.rect(p);
                        self.mc_rect(x, y, w, h);
                    }
                } else {
                    for p in 0..4 {
                        let sub = self.cur.sub[p];
                        let (x8, y8) = ((p & 1) * 2, (p >> 1) * 2);
                        if sub.direct {
                            for y in (0..2).step_by(direct_unit) {
                                for x in (0..2).step_by(direct_unit) {
                                    self.mc_rect(x8 + x, y8 + y, direct_unit, direct_unit);
                                }
                            }
                        } else {
                            for j in 0..sub.shape.num_parts() {
                                let (sx, sy, w, h) = sub.shape.rect(j);
                                self.mc_rect(x8 + sx, y8 + sy, w, h);
                            }
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // Intra prediction mode derivation (8.3.1.1 / 8.3.2.1)
    // ------------------------------------------------------------------------------------------

    fn intra_mode_nb(&self, n: usize, bx: usize, by: usize) -> Option<u8> {
        // returns Some(mode) or None if dcPredModePredictedFlag must be set
        let m = self.nb[n]?;
        let st = &self.pic.mbs[m];
        if self.pps.constrained_intra_pred && !st.kind.is_intra() {
            return None;
        }
        Some(match st.kind {
            MbKind::I4x4 | MbKind::I8x8 => st.intra_modes[by * 4 + bx],
            _ => 2,
        })
    }

    /// [`Self::intra_mode_nb`] for MBAFF frames: the block covering neighbouring luma location
    /// (xn, yn); an Intra_8x8 block next to Intra_4x4 blocks uses block n8 * 4 + 1 (A) / + 2 (B).
    fn intra_mode_mbaff(&self, xn: i32, yn: i32, cur8x8: bool, is_a: bool, b8: usize) -> Option<u8> {
        let (m, xw, yw) = self.nb_loc_mbaff(xn, yn, 16, 16)?;
        let st = &self.pic.mbs[m];
        if self.pps.constrained_intra_pred && !st.kind.is_intra() {
            return None;
        }
        let (xw, yw) = (xw as usize, yw as usize);
        Some(match st.kind {
            MbKind::I4x4 if cur8x8 => {
                let n8 = (yw / 8) * 2 + xw / 8;
                // 8.3.2.1: block n8 * 4 + n, n = 1 for A (3 for the lower left 8x8 block of a
                // frame macroblock next to a field pair), 2 for B
                let n = match (is_a, !self.field_mb && st.field && b8 == 2) {
                    (true, true) => 3,
                    (true, false) => 1,
                    (false, _) => 2,
                };
                let (bx, by) = BLK4_XY[n8 * 4 + n];
                st.intra_modes[by as usize * 4 + bx as usize]
            }
            MbKind::I4x4 | MbKind::I8x8 => st.intra_modes[(yw / 4) * 4 + xw / 4],
            _ => 2,
        })
    }

    /// Resolve Intra4x4PredMode / Intra8x8PredMode for the current MB from `self.cur.rem_mode`.
    pub fn derive_intra_modes(&mut self, is8x8: bool) {
        let step = if is8x8 { 2 } else { 1 };
        let nblk = if is8x8 { 4 } else { 16 };
        for i in 0..nblk {
            let (bx, by) = if is8x8 {
                ((i & 1) * 2, (i >> 1) * 2)
            } else {
                let (x, y) = BLK4_XY[i];
                (x as usize, y as usize)
            };
            let a = if bx > 0 {
                Some(self.mb().intra_modes[by * 4 + bx - 1])
            } else if self.mbaff {
                self.intra_mode_mbaff(-1, by as i32 * 4, is8x8, true, i)
            } else {
                self.intra_mode_nb(0, 3, by)
            };
            let b = if by > 0 {
                Some(self.mb().intra_modes[(by - 1) * 4 + bx])
            } else if self.mbaff {
                self.intra_mode_mbaff(bx as i32 * 4, -1, is8x8, false, i)
            } else {
                self.intra_mode_nb(1, bx, 3)
            };
            let pred = match (a, b) {
                (Some(a), Some(b)) => a.min(b),
                _ => 2,
            };
            let rem = self.cur.rem_mode[i];
            let mode = if rem < 0 {
                pred
            } else if (rem as u8) < pred {
                rem as u8
            } else {
                rem as u8 + 1
            };
            let st = self.mb_mut();
            for yy in by..by + step {
                for xx in bx..bx + step {
                    st.intra_modes[yy * 4 + xx] = mode;
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // Reconstruction
    // ------------------------------------------------------------------------------------------

    fn add_luma_block4(&mut self, raster: usize) {
        if self.s.blk_nz & (1 << raster) == 0 {
            return;
        }
        let stride = self.lstride;
        let o = self.lbase + (self.ly0 + (raster >> 2) * 4) * stride + self.lx0 + (raster & 3) * 4;
        let Some(blk) = self.s.coef.get_mut(raster * 16..).and_then(|c| c.first_chunk_mut::<16>()) else {
            return;
        };
        transform::idct4_add(blk, &mut self.pic.planes.y[o..], stride);
        blk.fill(0);
    }

    fn add_luma_block8(&mut self, b8: usize) {
        if self.s.blk_nz & (1 << b8) == 0 {
            return;
        }
        let stride = self.lstride;
        let o = self.lbase + (self.ly0 + (b8 >> 1) * 8) * stride + self.lx0 + (b8 & 1) * 8;
        let Some(blk) = self.s.coef.get_mut(b8 * 64..).and_then(|c| c.first_chunk_mut::<64>()) else {
            return;
        };
        transform::idct8_add(blk, &mut self.pic.planes.y[o..], stride);
        blk.fill(0);
    }

    fn add_chroma(&mut self) {
        let stride = self.cstride;
        for c in 0..2 {
            if self.s.c_nz[c] == 0 {
                continue;
            }
            for b in 0..4 {
                if self.s.c_nz[c] & (1 << b) == 0 {
                    continue;
                }
                let o = self.cbase + (self.cy0 + (b >> 1) * 4) * stride + self.cx0 + (b & 1) * 4;
                let Some(blk) = self.s.coef_c[c].get_mut(b * 16..).and_then(|c| c.first_chunk_mut::<16>()) else {
                    continue;
                };
                let plane = if c == 0 { &mut self.pic.planes.cb } else { &mut self.pic.planes.cr };
                transform::idct4_add(blk, &mut plane[o..], stride);
                blk.fill(0);
            }
            self.s.c_nz[c] = 0;
        }
    }

    /// MBAFF: the left neighbouring samples of rows y0..y0 + n all lie in macroblocks usable
    /// for intra prediction (a left pair of the other frame / field kind supplies the rows from
    /// both of its macroblocks).
    fn left_rows_intra(&self, y0: i32, n: i32) -> bool {
        let ok = |m: usize| !self.pps.constrained_intra_pred || self.pic.mbs[m].kind.is_intra();
        (y0..y0 + n).all(|y| self.nb_loc_mbaff(-1, y, 16, 16).is_some_and(|(m, _, _)| ok(m)))
    }

    /// Sample availability of the `n` x `n` luma block at 4x4 position (bx, by).
    fn luma_avail4(&self, bx: usize, by: usize, n: usize) -> Avail {
        let mixed_left = self.mbaff && bx == 0 && self.nb_left2.is_some();
        let a = if mixed_left { self.left_rows_intra(by as i32 * 4, n as i32) } else { self.intra_nb(0) };
        let b = self.intra_nb(1);
        let c = self.intra_nb(2);
        let d = self.intra_nb(3);
        let left = if bx > 0 { true } else { a };
        let top = if by > 0 { true } else { b };
        let top_left = match (bx > 0, by > 0) {
            (true, true) => true,
            (false, true) if mixed_left => self.left_rows_intra(by as i32 * 4 - 1, 1),
            (false, true) => a,
            (true, false) => b,
            (false, false) => d,
        };
        let top_right = if by == 0 {
            if bx < 3 { b } else { c }
        } else if bx == 3 {
            false
        } else {
            RASTER_TO_BLK4[(by - 1) * 4 + bx + 1] < RASTER_TO_BLK4[by * 4 + bx]
        };
        Avail { left, top, top_left, top_right }
    }

    /// Reconstruct an intra MB (prediction + residual) after its syntax has been parsed.
    pub fn reconstruct_intra(&mut self) {
        let kind = self.mb().kind;
        let stride = self.lstride;
        let (x0, y0) = (self.lx0, self.ly0);
        let lb = self.lbase;
        match kind {
            MbKind::I4x4 => {
                for (blk, &(bx, by)) in BLK4_XY.iter().enumerate() {
                    let (bx, by) = (bx as usize, by as usize);
                    let _ = blk;
                    let av = self.luma_avail4(bx, by, 4);
                    let mode = self.mb().intra_modes[by * 4 + bx];
                    intra::pred4x4(&mut self.pic.planes.y[lb..], stride, x0 + bx * 4, y0 + by * 4, mode, av);
                    self.add_luma_block4(by * 4 + bx);
                }
            }
            MbKind::I8x8 => {
                for b8 in 0..4 {
                    let (bx, by) = ((b8 & 1) * 2, (b8 >> 1) * 2);
                    let mut av = self.luma_avail4(bx, by, 8);
                    av.top_right = match b8 {
                        0 => self.intra_nb(1),
                        1 => self.intra_nb(2),
                        2 => true,
                        _ => false,
                    };
                    let mode = self.mb().intra_modes[by * 4 + bx];
                    intra::pred8x8(&mut self.pic.planes.y[lb..], stride, x0 + bx * 4, y0 + by * 4, mode, av);
                    self.add_luma_block8(b8);
                }
            }
            MbKind::I16x16 => {
                let av = Avail { left: self.intra_nb(0), top: self.intra_nb(1), top_left: self.intra_nb(3), top_right: false };
                intra::pred16x16(&mut self.pic.planes.y[lb..], stride, x0, y0, self.cur.info.i16_mode, av);
                for r in 0..16 {
                    self.add_luma_block4(r);
                }
            }
            _ => {}
        }
        let av = Avail { left: self.intra_nb(0), top: self.intra_nb(1), top_left: self.intra_nb(3), top_right: false };
        let cs = self.cstride;
        let mode = self.mb().intra_chroma_mode;
        let (cx, cy, cb) = (self.cx0, self.cy0, self.cbase);
        let halves = if self.mbaff && self.nb_left2.is_some() {
            let ok = |m: usize| !self.pps.constrained_intra_pred || self.pic.mbs[m].kind.is_intra();
            let half = |y0: i32| (y0..y0 + 4).all(|y| self.nb_loc_mbaff(-1, y, 8, 8).is_some_and(|(m, _, _)| ok(m)));
            [half(0), half(4)]
        } else {
            [av.left; 2]
        };
        intra::pred_chroma_halves(&mut self.pic.planes.cb[cb..], cs, cx, cy, mode, av, halves);
        intra::pred_chroma_halves(&mut self.pic.planes.cr[cb..], cs, cx, cy, mode, av, halves);
        self.add_chroma();
        self.s.blk_nz = 0;
    }

    /// Add the residual of an inter MB (prediction already written).
    pub fn reconstruct_inter_residual(&mut self) {
        if self.mb().transform_8x8 {
            for b8 in 0..4 {
                self.add_luma_block8(b8);
            }
        } else if self.s.blk_nz != 0 {
            for r in 0..16 {
                self.add_luma_block4(r);
            }
        }
        self.add_chroma();
        self.s.blk_nz = 0;
    }

    /// Write I_PCM samples.
    pub fn write_pcm(&mut self, samples: &[u8]) {
        let stride = self.lstride;
        let o = self.lbase + self.ly0 * stride + self.lx0;
        for r in 0..16 {
            self.pic.planes.y[o + r * stride..o + r * stride + 16].copy_from_slice(&samples[r * 16..r * 16 + 16]);
        }
        let cs = self.cstride;
        let co = self.cbase + self.cy0 * cs + self.cx0;
        for c in 0..2 {
            let src = &samples[256 + c * 64..256 + c * 64 + 64];
            let plane = if c == 0 { &mut self.pic.planes.cb } else { &mut self.pic.planes.cr };
            for r in 0..8 {
                plane[co + r * cs..co + r * cs + 8].copy_from_slice(&src[r * 8..r * 8 + 8]);
            }
        }
    }

    /// Common state for an I_PCM MB.
    pub fn finish_pcm(&mut self) {
        let qpc = [self.chroma_qp(0, 0) as u8, self.chroma_qp(0, 1) as u8];
        let st = self.mb_mut();
        st.kind = MbKind::IPcm;
        st.qp = 0;
        st.qpc = qpc;
        st.nnz = [16; 16];
        st.nnz_c = [[16; 4]; 2];
        st.cbf_dc = 7;
        st.cbp = 0x2f;
        st.nz_mask = 0xffff;
        st.ref_idx = [[-1; 4]; 2];
    }

    /// Update QP after mb_qp_delta.
    pub fn apply_qp_delta(&mut self, delta: i32) -> Result<()> {
        ensure!((-26..=25).contains(&delta), "mb_qp_delta {delta} out of range");
        self.qp = (self.qp + delta + 52) % 52;
        Ok(())
    }

    pub fn store_qp(&mut self) {
        let qp = self.qp;
        let qpc = [self.chroma_qp(qp, 0) as u8, self.chroma_qp(qp, 1) as u8];
        let st = self.mb_mut();
        st.qp = qp as u8;
        st.qpc = qpc;
    }

    // ------------------------------------------------------------------------------------------
    // Residual storage helpers (shared by CAVLC and CABAC)
    // ------------------------------------------------------------------------------------------

    fn luma_list4(&self) -> usize {
        if self.mb().kind.is_intra() { 0 } else { 3 }
    }

    /// Store a 4x4 luma block given in scan order (levels[0..16]); `ac_only` skips position 0.
    pub fn put_luma4(&mut self, raster: usize, levels: &[i32; 16], ac_only: bool) {
        let qp = self.qp;
        let ls = &self.ls.ls4[self.luma_list4()][(qp % 6) as usize];
        let blk = &mut self.s.coef[raster * 16..raster * 16 + 16];
        let mut any = false;
        for (k, &lv) in levels.iter().enumerate().skip(ac_only as usize) {
            if lv != 0 {
                let p = self.scan4[k] as usize;
                blk[p] = transform::scale4(lv, ls[p], qp);
                any = true;
            }
        }
        if any {
            self.s.blk_nz |= 1 << raster;
            self.pic.mbs[self.mb_addr].nz_mask |= 1 << raster;
        }
    }

    /// Store an 8x8 luma block given in scan order.
    pub fn put_luma8(&mut self, b8: usize, levels: &[i32; 64]) {
        let qp = self.qp;
        let list = if self.mb().kind.is_intra() { 0 } else { 1 };
        let ls = &self.ls.ls8[list][(qp % 6) as usize];
        let blk = &mut self.s.coef[b8 * 64..b8 * 64 + 64];
        let mut any = false;
        for (k, &lv) in levels.iter().enumerate() {
            if lv != 0 {
                let p = self.scan8[k] as usize;
                blk[p] = transform::scale8(lv, ls[p], qp);
                any = true;
            }
        }
        if any {
            self.s.blk_nz |= 1 << b8;
            let (bx, by) = ((b8 & 1) * 2, (b8 >> 1) * 2);
            let m = (0b11u16 << (by * 4 + bx)) | (0b11u16 << ((by + 1) * 4 + bx));
            self.pic.mbs[self.mb_addr].nz_mask |= m;
        }
    }

    /// Intra16x16 DC levels in scan order -> scaled DC of each 4x4 block.
    pub fn put_luma_dc(&mut self, levels: &[i32; 16]) {
        let mut c = [0i32; 16];
        for (k, &lv) in levels.iter().enumerate() {
            c[self.scan4[k] as usize] = lv;
        }
        let qp = self.qp;
        let ls00 = self.ls.ls4[0][(qp % 6) as usize][0];
        transform::luma_dc_dequant(&mut c, qp, ls00);
        for (r, &dc) in c.iter().enumerate() {
            if dc != 0 {
                self.s.coef[r * 16] = dc;
                self.s.blk_nz |= 1 << r;
            }
        }
    }

    /// Chroma DC levels (4:2:0: 4 values in raster order) for component `c`.
    pub fn put_chroma_dc(&mut self, c: usize, levels: &[i32; 4]) {
        let mut v = *levels;
        let qpc = self.chroma_qp(self.qp, c);
        let list = if self.mb().kind.is_intra() { 1 + c } else { 4 + c };
        let ls00 = self.ls.ls4[list][(qpc % 6) as usize][0];
        transform::chroma_dc_dequant_420(&mut v, qpc, ls00);
        for (b, &dc) in v.iter().enumerate() {
            if dc != 0 {
                self.s.coef_c[c][b * 16] = dc;
                self.s.c_nz[c] |= 1 << b;
            }
        }
    }

    /// Chroma AC levels (scan positions 1..16 in levels[1..]) for block `b` of component `c`.
    pub fn put_chroma_ac(&mut self, c: usize, b: usize, levels: &[i32; 16]) {
        let qpc = self.chroma_qp(self.qp, c);
        let list = if self.mb().kind.is_intra() { 1 + c } else { 4 + c };
        let ls = &self.ls.ls4[list][(qpc % 6) as usize];
        let blk = &mut self.s.coef_c[c][b * 16..b * 16 + 16];
        let mut any = false;
        for (k, &lv) in levels.iter().enumerate().skip(1) {
            if lv != 0 {
                let p = self.scan4[k] as usize;
                blk[p] = transform::scale4(lv, ls[p], qpc);
                any = true;
            }
        }
        if any {
            self.s.c_nz[c] |= 1 << b;
        }
    }

    // ------------------------------------------------------------------------------------------
    // CAVLC macroblock layer
    // ------------------------------------------------------------------------------------------

    /// Decode a P_Skip or B_Skip macroblock at the current address.
    pub fn decode_skip(&mut self) -> Result<()> {
        let b = self.sh.slice_type.is_b();
        self.cur = MbCur::default();
        self.cur.info.kind = if b { MbKind::BSkip } else { MbKind::PSkip };
        self.mb_mut().kind = self.cur.info.kind;
        self.store_qp();
        if b {
            self.derive_direct(0xf)?;
        } else {
            ensure!(!self.refs[0].is_empty(), "P_Skip with empty reference list");
            self.derive_p_skip();
        }
        self.predict_inter();
        self.prev_qp_delta_nz = false;
        Ok(())
    }

    fn read_te(r: &mut BitReader, range: u32) -> Result<u32> {
        Ok(if range > 1 { r.read_ue()? } else { !r.read_bit()? as u32 })
    }

    /// Parse and reconstruct one macroblock_layer() with CAVLC.
    pub fn decode_mb_cavlc(&mut self, r: &mut BitReader) -> Result<()> {
        let mb_type = r.read_ue()?;
        let st = self.sh.slice_type;
        let info = match st {
            SliceType::I | SliceType::Si => intra_mb_type(mb_type),
            SliceType::P | SliceType::Sp => {
                if mb_type < 5 {
                    p_mb_type(mb_type)
                } else {
                    intra_mb_type(mb_type - 5)
                }
            }
            SliceType::B => {
                if mb_type < 23 {
                    b_mb_type(mb_type)
                } else {
                    intra_mb_type(mb_type - 23)
                }
            }
        };
        let Some(mut info) = info else { return invalid(format!("mb_type {mb_type} out of range")) };
        self.cur = MbCur { info, ..Default::default() };
        if info.kind == MbKind::IPcm {
            r.byte_align();
            let pos = r.byte_pos();
            let data = r.data();
            ensure!(data.len() >= pos + 384, "truncated I_PCM macroblock");
            let Some(&samples) = data.get(pos..).and_then(|d| d.first_chunk::<384>()) else {
                return invalid("truncated I_PCM macroblock");
            };
            r.skip(384 * 8)?;
            self.finish_pcm();
            self.write_pcm(&samples);
            return Ok(());
        }
        let mut transform_8x8 = false;
        if info.kind == MbKind::I4x4 && self.pps.transform_8x8_mode && r.read_flag()? {
            transform_8x8 = true;
            info.kind = MbKind::I8x8;
            self.cur.info.kind = MbKind::I8x8;
        }
        self.mb_mut().kind = info.kind;
        self.mb_mut().transform_8x8 = transform_8x8;
        // field macroblocks of MBAFF frames index field lists of twice the length (7.4.5.1)
        let fm = if self.mbaff && self.field_mb { 2 } else { 1 };
        let n_ref = [self.sh.num_ref_idx_active[0] * fm, self.sh.num_ref_idx_active[1] * fm];
        match info.kind {
            MbKind::I4x4 | MbKind::I8x8 => {
                let n = if info.kind == MbKind::I8x8 { 4 } else { 16 };
                for i in 0..n {
                    self.cur.rem_mode[i] = if r.read_flag()? { -1 } else { r.read_bits(3)? as i8 };
                }
                self.derive_intra_modes(info.kind == MbKind::I8x8);
                let m = r.read_ue()?;
                ensure!(m <= 3, "intra_chroma_pred_mode out of range");
                self.mb_mut().intra_chroma_mode = m as u8;
            }
            MbKind::I16x16 => {
                let m = r.read_ue()?;
                ensure!(m <= 3, "intra_chroma_pred_mode out of range");
                self.mb_mut().intra_chroma_mode = m as u8;
            }
            MbKind::BDirect16x16 => {}
            _ => {
                if info.part == Part::P8x8 {
                    for p in 0..4 {
                        let t = r.read_ue()?;
                        let sub = if st.is_b() { b_sub_mb_type(t) } else { p_sub_mb_type(t) };
                        let Some(sub) = sub else { return invalid(format!("sub_mb_type {t} out of range")) };
                        self.cur.sub[p] = sub;
                    }
                    for l in 0..2 {
                        for p in 0..4 {
                            let sub = self.cur.sub[p];
                            if sub.direct || sub.pred & (1 << l) == 0 {
                                continue;
                            }
                            self.cur.ref_idx[l][p] = if n_ref[l] > 1 && !info.ref0 { Self::read_te(r, n_ref[l] - 1)? as i8 } else { 0 };
                        }
                    }
                    for l in 0..2 {
                        for p in 0..4 {
                            let sub = self.cur.sub[p];
                            if sub.direct || sub.pred & (1 << l) == 0 {
                                continue;
                            }
                            for j in 0..sub.shape.num_parts() {
                                let x = r.read_se()?;
                                let y = r.read_se()?;
                                self.cur.mvd[l][p * 4 + j] = [x as i16, y as i16];
                            }
                        }
                    }
                } else {
                    let np = info.part.num_parts();
                    for l in 0..2 {
                        for p in 0..np {
                            if info.pred[p] & (1 << l) != 0 {
                                self.cur.ref_idx[l][p] = if n_ref[l] > 1 { Self::read_te(r, n_ref[l] - 1)? as i8 } else { 0 };
                            }
                        }
                    }
                    for l in 0..2 {
                        for p in 0..np {
                            if info.pred[p] & (1 << l) != 0 {
                                let x = r.read_se()?;
                                let y = r.read_se()?;
                                self.cur.mvd[l][p * 4] = [x as i16, y as i16];
                            }
                        }
                    }
                }
            }
        }
        // coded_block_pattern
        let cbp = if info.kind == MbKind::I16x16 {
            info.i16_cbp
        } else {
            let code = r.read_ue()?;
            ensure!(code < 48, "coded_block_pattern out of range");
            let (i, p) = CBP_ME[code as usize];
            if info.kind.is_intra() { i } else { p }
        };
        self.cur.cbp = cbp;
        self.mb_mut().cbp = cbp;
        if !info.kind.is_intra() && cbp & 15 != 0 && self.pps.transform_8x8_mode && self.no_sub_8x8_lt() {
            let t = r.read_flag()?;
            self.mb_mut().transform_8x8 = t;
        }
        if !info.kind.is_intra() {
            self.derive_inter_motion()?;
        }
        if cbp != 0 || info.kind == MbKind::I16x16 {
            let d = r.read_se()?;
            self.apply_qp_delta(d)?;
        }
        self.store_qp();
        self.residual_cavlc(r, cbp)?;
        if info.kind.is_intra() {
            self.reconstruct_intra();
        } else {
            self.predict_inter();
            self.reconstruct_inter_residual();
        }
        Ok(())
    }

    /// Whether transform_size_8x8_flag may be present for an inter MB (NoSubMbPartSizeLessThan8x8 etc.).
    pub fn no_sub_8x8_lt(&self) -> bool {
        let info = self.cur.info;
        match info.kind {
            MbKind::BDirect16x16 => self.sps.direct_8x8_inference,
            MbKind::Inter if info.part == Part::P8x8 => {
                self.cur.sub.iter().all(|s| if s.direct { self.sps.direct_8x8_inference } else { s.shape == SubPart::S8x8 })
            }
            _ => true,
        }
    }

    fn residual_cavlc(&mut self, r: &mut BitReader, cbp: u8) -> Result<()> {
        let kind = self.mb().kind;
        let t8 = self.mb().transform_8x8;
        if kind == MbKind::I16x16 {
            let nc = self.nc_luma(0);
            let mut lv = [0i32; 16];
            cavlc::residual_block(r, &mut lv, 0, 15, 16, nc)?;
            self.put_luma_dc(&lv);
        }
        for b8 in 0..4 {
            let (bx8, by8) = ((b8 & 1) * 2, (b8 >> 1) * 2);
            if cbp & (1 << b8) == 0 {
                continue;
            }
            if t8 {
                let mut lv8 = [0i32; 64];
                for i4 in 0..4 {
                    let raster = (by8 + (i4 >> 1)) * 4 + bx8 + (i4 & 1);
                    let nc = self.nc_luma(raster);
                    let mut lv = [0i32; 16];
                    let tc = cavlc::residual_block(r, &mut lv, 0, 15, 16, nc)?;
                    self.mb_mut().nnz[raster] = tc;
                    for (i, &v) in lv.iter().enumerate() {
                        lv8[4 * i + i4] = v;
                    }
                }
                self.put_luma8(b8, &lv8);
            } else {
                for i4 in 0..4 {
                    let raster = (by8 + (i4 >> 1)) * 4 + bx8 + (i4 & 1);
                    let nc = self.nc_luma(raster);
                    let mut lv = [0i32; 16];
                    let tc = if kind == MbKind::I16x16 {
                        cavlc::residual_block(r, &mut lv[1..], 0, 14, 15, nc)?
                    } else {
                        cavlc::residual_block(r, &mut lv, 0, 15, 16, nc)?
                    };
                    self.mb_mut().nnz[raster] = tc;
                    self.put_luma4(raster, &lv, kind == MbKind::I16x16);
                }
            }
        }
        let cbp_c = cbp >> 4;
        if cbp_c & 3 != 0 {
            for c in 0..2 {
                let mut lv = [0i32; 4];
                cavlc::residual_block(r, &mut lv, 0, 3, 4, -1)?;
                self.put_chroma_dc(c, &lv);
            }
        }
        if cbp_c & 2 != 0 {
            for c in 0..2 {
                for b in 0..4 {
                    let nc = self.nc_chroma(c, b);
                    let mut lv = [0i32; 16];
                    let tc = cavlc::residual_block(r, &mut lv[1..], 0, 14, 15, nc)?;
                    self.mb_mut().nnz_c[c][b] = tc;
                    self.put_chroma_ac(c, b, &lv);
                }
            }
        }
        Ok(())
    }

    /// Decode slice_data() with CAVLC. `r` is positioned after the slice header.
    pub fn decode_cavlc(&mut self, r: &mut BitReader) -> Result<()> {
        if self.mbaff {
            return self.decode_cavlc_mbaff(r);
        }
        let total = self.mb_w * self.mb_h;
        let mut addr = self.sh.first_mb_in_slice as usize;
        let intra_slice = self.sh.slice_type.is_intra();
        loop {
            ensure!(addr < total, "macroblock address {addr} out of range");
            if !intra_slice {
                let run = r.read_ue()? as usize;
                ensure!(addr + run <= total, "mb_skip_run out of range");
                for _ in 0..run {
                    self.start_mb(addr);
                    self.decode_skip()?;
                    self.pic.mb_done(addr);
                    addr += 1;
                }
                if run > 0 && !r.more_rbsp_data() {
                    break;
                }
                ensure!(addr < total, "macroblock address {addr} out of range");
            }
            self.start_mb(addr);
            self.decode_mb_cavlc(r)?;
            self.pic.mb_done(addr);
            addr += 1;
            if !r.more_rbsp_data() {
                break;
            }
        }
        Ok(())
    }
}
