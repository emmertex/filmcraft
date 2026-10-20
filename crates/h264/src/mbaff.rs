//! Macroblock-adaptive frame/field (MBAFF) frames: macroblock pair addressing, neighbour
//! derivation (6.4.12.2), mb_field_decoding_flag inference (7.4.4) and the slice data loops
//! (7.3.4) for CAVLC and CABAC.

use crate::cabac::Cabac;
use crate::error::{Result, ensure};
use crate::picture::{MbKind, MbState};
use crate::slicedec::SliceDecoder;
use crate::tables::{FIELD4, FIELD8, ZIGZAG4, ZIGZAG8};
use filmcraft_bitstream::BitReader;

impl SliceDecoder<'_> {
    /// Set up macroblock `addr` (MBAFF address: 2 * pair + bottom) as a frame or field macroblock.
    pub fn start_mb_mbaff(&mut self, addr: usize, field: bool) {
        let w = self.mb_w;
        let pair = addr / 2;
        let bottom = addr & 1;
        let (mx, py) = (pair % w, pair / w);
        let my = 2 * py + bottom;
        let idx = my * w + mx;
        self.cur_addr = addr;
        self.mb_addr = idx;
        self.mb_x = mx;
        self.mb_y = my;
        self.field_mb = field;
        self.parity = bottom as u8;
        let (pw, cw) = (self.pic.planes.width, self.pic.planes.cwidth);
        if field {
            // a field macroblock covers every second line of the pair, starting at line `bottom`
            self.lbase = bottom * pw;
            self.lstride = 2 * pw;
            self.ly0 = py * 16;
            self.cbase = bottom * cw;
            self.cstride = 2 * cw;
            self.cy0 = py * 8;
            self.scan4 = &FIELD4;
            self.scan8 = &FIELD8;
        } else {
            self.lbase = 0;
            self.lstride = pw;
            self.ly0 = my * 16;
            self.cbase = 0;
            self.cstride = cw;
            self.cy0 = my * 8;
            self.scan4 = &ZIGZAG4;
            self.scan8 = &ZIGZAG8;
        }
        self.lx0 = mx * 16;
        self.cx0 = mx * 8;
        self.mv_done = 0;
        let sn = self.slice_num;
        if let Some(st) = self.pic.mbs.get_mut(idx) {
            *st = MbState { slice_num: sn, field, ..MbState::default() };
        }
        let mb = |n: Option<(usize, i32, i32)>| n.map(|(m, _, _)| m);
        self.nb = [
            mb(self.nb_loc_mbaff(-1, 0, 16, 16)),
            mb(self.nb_loc_mbaff(0, -1, 16, 16)),
            mb(self.nb_loc_mbaff(16, -1, 16, 16)),
            mb(self.nb_loc_mbaff(-1, -1, 16, 16)),
        ];
        // the left column of a frame / field macroblock next to a field / frame pair comes from both
        // macroblocks of that pair
        self.nb_left2 = mb(self.nb_loc_mbaff(-1, 15, 16, 16)).filter(|&m| Some(m) != self.nb[0]);
    }

    /// Macroblock pair (dx, dy) pairs away from the current one, if available (decoded in the
    /// current slice): (index of its top macroblock, field pair).
    fn pair_at(&self, dx: i32, dy: i32) -> Option<(usize, bool)> {
        let w = self.mb_w as i32;
        let x = self.mb_x as i32 + dx;
        let y = (self.mb_y / 2) as i32 + dy;
        if x < 0 || x >= w || y < 0 {
            return None;
        }
        let t = (2 * y * w + x) as usize;
        let st = self.pic.mbs.get(t)?;
        (st.slice_num == self.slice_num).then_some((t, st.field))
    }

    /// Inferred mb_field_decoding_flag of the current pair when it is not coded for either
    /// macroblock (7.4.4): from the left pair, else the pair above (same slice), else frame.
    pub fn infer_field_mbaff(&self, addr: usize) -> bool {
        let w = self.mb_w;
        let pair = addr / 2;
        let (mx, py) = (pair % w, pair / w);
        let sn = self.slice_num;
        let at = |x: usize, y: usize| self.pic.mbs.get(2 * y * w + x).filter(|st| st.slice_num == sn).map(|st| st.field);
        if mx > 0
            && let Some(f) = at(mx - 1, py)
        {
            return f;
        }
        if py > 0
            && let Some(f) = at(mx, py - 1)
        {
            return f;
        }
        false
    }

    /// Neighbouring location (xn, yn) relative to the current macroblock of an MBAFF frame
    /// (6.4.12.2, Table 6-4) for a `max_w` x `max_h` component: the macroblock (index into
    /// `pic.mbs`) and the location (xW, yW) inside it.
    pub fn nb_loc_mbaff(&self, xn: i32, yn: i32, max_w: i32, max_h: i32) -> Option<(usize, i32, i32)> {
        if (0..max_w).contains(&xn) && (0..max_h).contains(&yn) {
            return Some((self.mb_addr, xn, yn));
        }
        let w = self.mb_w;
        let top = self.mb_y.is_multiple_of(2);
        let cur_frame = !self.field_mb;
        let bot = |t: usize| t + w;
        let (n, ym) = if xn < 0 && yn < 0 {
            // D
            if cur_frame {
                if top {
                    let (t, _) = self.pair_at(-1, -1)?;
                    (bot(t), yn)
                } else {
                    // the line above the bottom frame macroblock: line 15 of the left pair, in its
                    // bottom field macroblock (row 7) when that pair is a field pair
                    let (t, f) = self.pair_at(-1, 0)?;
                    if f { (bot(t), (yn + max_h) >> 1) } else { (t, yn) }
                }
            } else if top {
                let (t, f) = self.pair_at(-1, -1)?;
                if f { (t, yn) } else { (bot(t), 2 * yn) }
            } else {
                let (t, _) = self.pair_at(-1, -1)?;
                (bot(t), yn)
            }
        } else if xn < 0 && yn < max_h {
            // A
            let (t, f) = self.pair_at(-1, 0)?;
            match (cur_frame, top, f) {
                (true, true, false) => (t, yn),
                (true, true, true) => (if yn % 2 == 0 { t } else { bot(t) }, yn >> 1),
                (true, false, false) => (bot(t), yn),
                (true, false, true) => (if yn % 2 == 0 { t } else { bot(t) }, (yn + max_h) >> 1),
                (false, true, false) => {
                    if yn < max_h / 2 {
                        (t, yn << 1)
                    } else {
                        (bot(t), (yn << 1) - max_h)
                    }
                }
                (false, true, true) => (t, yn),
                (false, false, false) => {
                    if yn < max_h / 2 {
                        (t, (yn << 1) + 1)
                    } else {
                        (bot(t), (yn << 1) + 1 - max_h)
                    }
                }
                (false, false, true) => (bot(t), yn),
            }
        } else if xn >= 0 && xn < max_w && yn < 0 {
            // B
            if cur_frame {
                if top {
                    let (t, _) = self.pair_at(0, -1)?;
                    (bot(t), yn)
                } else {
                    // the top macroblock of the same pair
                    (self.mb_addr - w, yn)
                }
            } else if top {
                let (t, f) = self.pair_at(0, -1)?;
                if f { (t, yn) } else { (bot(t), 2 * yn) }
            } else {
                let (t, _) = self.pair_at(0, -1)?;
                (bot(t), yn)
            }
        } else if xn >= max_w && yn < 0 {
            // C
            if cur_frame {
                if !top {
                    return None;
                }
                let (t, _) = self.pair_at(1, -1)?;
                (bot(t), yn)
            } else if top {
                let (t, f) = self.pair_at(1, -1)?;
                if f { (t, yn) } else { (bot(t), 2 * yn) }
            } else {
                let (t, _) = self.pair_at(1, -1)?;
                (bot(t), yn)
            }
        } else {
            return None;
        };
        Some((n, xn.rem_euclid(max_w), ym.rem_euclid(max_h)))
    }

    /// Neighbouring luma 4x4 block (4x4 units relative to the current macroblock, outside it):
    /// (macroblock index, raster index of the 4x4 block in it).
    #[inline]
    pub fn nb_blk4_mbaff(&self, x: i32, y: i32) -> Option<(usize, usize)> {
        // the luma location of the block's top-left sample, or the sample just outside
        let c = |v: i32| if v < 0 { -1 } else { v * 4 };
        let (m, xw, yw) = self.nb_loc_mbaff(c(x), c(y), 16, 16)?;
        Some((m, ((yw >> 2) * 4 + (xw >> 2)) as usize))
    }

    /// Neighbouring chroma 4x4 block (4:2:0, 2x2 blocks): (macroblock index, raster index 0..4).
    #[inline]
    pub fn nb_chroma_blk_mbaff(&self, x: i32, y: i32) -> Option<(usize, usize)> {
        let c = |v: i32| if v < 0 { -1 } else { v * 4 };
        let (m, xw, yw) = self.nb_loc_mbaff(c(x), c(y), 8, 8)?;
        Some((m, ((yw >> 2) * 2 + (xw >> 2)) as usize))
    }

    /// Vertical motion vector / reference index of neighbour `st` seen from the current macroblock
    /// (8.4.1.3.1: frame and field macroblocks of an MBAFF frame).
    #[inline]
    pub fn mbaff_scale(&self, st: &MbState, ref_idx: i8, mv: [i16; 2]) -> (i8, [i16; 2]) {
        if ref_idx < 0 || st.field == self.field_mb {
            return (ref_idx, mv);
        }
        if self.field_mb { (ref_idx * 2, [mv[0], mv[1] / 2]) } else { (ref_idx >> 1, [mv[0], mv[1].wrapping_mul(2)]) }
    }

    /// slice_data() of an MBAFF frame with CAVLC (7.3.4).
    pub fn decode_cavlc_mbaff(&mut self, r: &mut BitReader) -> Result<()> {
        let total = self.mb_w * self.mb_h;
        let mut addr = self.sh.first_mb_in_slice as usize * 2;
        let intra_slice = self.sh.slice_type.is_intra();
        let mut field = false;
        // mb_field_decoding_flag of the current pair was already read (with a skipped top MB)
        let mut flag_read = false;
        loop {
            ensure!(addr < total, "macroblock address {addr} out of range");
            let mut prev_skipped = false;
            if !intra_slice {
                let run = r.read_ue()? as usize;
                ensure!(addr + run <= total, "mb_skip_run out of range");
                prev_skipped = run > 0;
                for i in 0..run {
                    if addr.is_multiple_of(2) {
                        // the flag of a pair whose bottom macroblock is coded follows the run
                        flag_read = i + 1 == run && r.more_rbsp_data();
                        field = if flag_read { r.read_flag()? } else { self.infer_field_mbaff(addr) };
                    }
                    self.start_mb_mbaff(addr, field);
                    self.decode_skip()?;
                    self.pic.mb_done(self.mb_addr);
                    addr += 1;
                }
                if run > 0 && !r.more_rbsp_data() {
                    break;
                }
                ensure!(addr < total, "macroblock address {addr} out of range");
            }
            // the pair's flag precedes its top macroblock, or a bottom one after a skipped top
            if addr.is_multiple_of(2) || (prev_skipped && !flag_read) {
                field = r.read_flag()?;
            }
            flag_read = false;
            self.start_mb_mbaff(addr, field);
            self.decode_mb_cavlc(r)?;
            self.pic.mb_done(self.mb_addr);
            addr += 1;
            if !r.more_rbsp_data() {
                break;
            }
        }
        Ok(())
    }

    /// mb_field_decoding_flag (ctxIdx 70..72).
    fn decode_field_flag(&self, c: &mut Cabac) -> bool {
        let cond = |dx: i32, dy: i32| self.pair_at(dx, dy).is_some_and(|(_, f)| f) as usize;
        c.decode_decision(70 + cond(-1, 0) + cond(0, -1)) == 1
    }

    /// slice_data() of an MBAFF frame with CABAC (7.3.4).
    pub fn decode_cabac_mbaff(&mut self, c: &mut Cabac) -> Result<()> {
        let total = self.mb_w * self.mb_h;
        let mut addr = self.sh.first_mb_in_slice as usize * 2;
        let intra_slice = self.sh.slice_type.is_intra();
        let skip_kind = if self.sh.slice_type.is_b() { MbKind::BSkip } else { MbKind::PSkip };
        let mut field = false;
        // skip flag of a bottom macroblock decoded ahead (with its skipped top macroblock)
        let mut bottom_skip: Option<bool> = None;
        loop {
            ensure!(addr < total, "macroblock address {addr} out of range");
            let top = addr.is_multiple_of(2);
            let skip = if intra_slice {
                if top {
                    self.start_mb_mbaff(addr, false);
                    field = self.decode_field_flag(c);
                }
                false
            } else if top {
                let inferred = self.infer_field_mbaff(addr);
                self.start_mb_mbaff(addr, inferred);
                if self.decode_skip_flag(c) {
                    // the bottom macroblock's skip flag (and its field flag) come next and decide
                    // the pair's field decoding of the skipped top macroblock
                    self.mb_mut().kind = skip_kind;
                    self.start_mb_mbaff(addr + 1, inferred);
                    let bs = self.decode_skip_flag(c);
                    field = if bs { inferred } else { self.decode_field_flag(c) };
                    bottom_skip = Some(bs);
                    true
                } else {
                    field = self.decode_field_flag(c);
                    false
                }
            } else {
                match bottom_skip.take() {
                    Some(bs) => bs,
                    None => {
                        self.start_mb_mbaff(addr, field);
                        self.decode_skip_flag(c)
                    }
                }
            };
            self.start_mb_mbaff(addr, field);
            if skip {
                self.decode_skip()?;
            } else {
                self.decode_mb_cabac(c)?;
            }
            self.pic.mb_done(self.mb_addr);
            ensure!(!c.overrun(), "CABAC read past end of slice data");
            if !top && c.decode_terminate() == 1 {
                break;
            }
            addr += 1;
        }
        Ok(())
    }
}
