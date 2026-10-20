//! Picture buffers and per-macroblock state.

use std::sync::{Arc, OnceLock};

/// Planar 4:2:0 8-bit sample buffers (MB-aligned dimensions, stride == width).
#[derive(Clone)]
pub struct Planes {
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub cwidth: usize,
}

impl Planes {
    pub fn new(width: usize, height: usize) -> Self {
        let (cw, ch) = (width / 2, height / 2);
        Planes { y: vec![0; width * height], cb: vec![128; cw * ch], cr: vec![128; cw * ch], width, height, cwidth: cw }
    }
    pub fn gray(width: usize, height: usize) -> Self {
        let mut p = Self::new(width, height);
        p.y.fill(128);
        p
    }
}

/// Macroblock coding category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MbKind {
    #[default]
    None,
    I4x4,
    I8x8,
    I16x16,
    IPcm,
    PSkip,
    BSkip,
    BDirect16x16,
    /// Any other P or B inter macroblock.
    Inter,
}

impl MbKind {
    #[inline]
    pub fn is_intra(self) -> bool {
        matches!(self, MbKind::I4x4 | MbKind::I8x8 | MbKind::I16x16 | MbKind::IPcm)
    }
}

/// Per-macroblock state kept for the whole picture (neighbour derivations, deblocking, co-located data).
#[derive(Clone, Copy)]
pub struct MbState {
    /// Slice number within the picture; `u32::MAX` = not (yet) decoded.
    pub slice_num: u32,
    pub kind: MbKind,
    pub transform_8x8: bool,
    /// coded_block_pattern: bits 0..3 luma, bits 4..5 chroma.
    pub cbp: u8,
    pub qp: u8,
    /// QPc for Cb / Cr (used by deblocking).
    pub qpc: [u8; 2],
    pub intra_chroma_mode: u8,
    /// Intra4x4PredMode / Intra8x8PredMode per 4x4 block (raster order).
    pub intra_modes: [u8; 16],
    /// CAVLC: TotalCoeff per luma 4x4 block (raster). CABAC: coded_block_flag.
    pub nnz: [u8; 16],
    /// Same for chroma AC blocks [Cb, Cr][raster 2x2].
    pub nnz_c: [[u8; 4]; 2],
    /// CABAC coded_block_flag of DC blocks: bit0 luma (I16x16), bit1 Cb, bit2 Cr.
    pub cbf_dc: u8,
    /// Luma 4x4 blocks (raster bit) with non-zero coefficients, for deblocking bS = 2.
    pub nz_mask: u16,
    pub ref_idx: [[i8; 4]; 2],
    /// Motion vectors per 4x4 block (raster order).
    pub mv: [[[i16; 2]; 16]; 2],
    /// |mvd| per 4x4 block (raster), saturated, for CABAC context selection.
    pub mvd: [[[u8; 2]; 16]; 2],
    /// Bit per 8x8 block predicted in direct mode.
    pub direct8x8: u8,
    /// Field macroblock (every macroblock of a field picture; field pairs of an MBAFF frame).
    pub field: bool,
}

impl MbState {
    #[inline]
    pub fn kind_is_skip(&self) -> bool {
        matches!(self.kind, MbKind::PSkip | MbKind::BSkip)
    }
}

impl Default for MbState {
    fn default() -> Self {
        MbState {
            slice_num: u32::MAX,
            kind: MbKind::None,
            transform_8x8: false,
            cbp: 0,
            qp: 0,
            qpc: [0; 2],
            intra_chroma_mode: 0,
            intra_modes: [2; 16],
            nnz: [0; 16],
            nnz_c: [[0; 4]; 2],
            cbf_dc: 0,
            nz_mask: 0,
            ref_idx: [[-1; 4]; 2],
            mv: [[[0; 2]; 16]; 2],
            mvd: [[[0; 2]; 16]; 2],
            direct8x8: 0,
            field: false,
        }
    }
}

/// One published macroblock row of a decoded frame: final (deblocked) samples plus the motion data
/// needed as co-located information for direct prediction.
pub struct FrameRow {
    /// 16 luma lines.
    pub y: Box<[u8]>,
    /// 8 lines per chroma component.
    pub cb: Box<[u8]>,
    pub cr: Box<[u8]>,
    /// Per 4x4 block: index mb_x * 16 + raster.
    pub mv: [Box<[[i16; 2]]>; 2],
    /// Per 8x8 block: index mb_x * 4 + b8.
    pub ref_idx: [Box<[i8]>; 2],
    /// Unique id of the referenced picture per 8x8 block (u32::MAX = none).
    pub ref_id: [Box<[u32]>; 2],
    pub intra: Box<[bool]>,
}

/// Motion data of one macroblock of a decoded picture, kept for direct prediction of later
/// pictures in interlaced streams (8.4.1.2.1).
#[derive(Clone, Copy)]
pub struct ColMb {
    pub intra: bool,
    /// Field macroblock (MBAFF field pair, or any macroblock of a field picture).
    pub field: bool,
    pub ref_idx: [[i8; 4]; 2],
    /// Identity ([`RefPic::id`]) of the referenced picture per 8x8 block (u32::MAX = none).
    pub ref_id: [[u32; 4]; 2],
    pub mv: [[[i16; 2]; 16]; 2],
}

impl Default for ColMb {
    fn default() -> Self {
        ColMb { intra: true, field: false, ref_idx: [[-1; 4]; 2], ref_id: [[u32::MAX; 4]; 2], mv: [[[0; 2]; 16]; 2] }
    }
}

/// How a picture of an interlaced stream was coded (PicCodingStruct, 8.4.1.2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColKind {
    /// Frame picture without MBAFF.
    Frm,
    /// MBAFF frame.
    Afrm,
    /// Field picture.
    Fld,
}

/// Motion data of a whole decoded picture (frame, or one field) of an interlaced stream.
pub struct ColPic {
    pub kind: ColKind,
    pub mb_w: usize,
    /// Macroblock rows of the coded picture (field rows for a field).
    pub mb_h: usize,
    /// Macroblocks in raster order of the picture's MB grid (for MBAFF: physical position, the
    /// top macroblock of a pair at row 2y, the bottom one at 2y + 1).
    pub mbs: Box<[ColMb]>,
}

impl ColPic {
    pub fn intra(kind: ColKind, mb_w: usize, mb_h: usize) -> Self {
        ColPic { kind, mb_w, mb_h, mbs: vec![ColMb::default(); mb_w * mb_h].into() }
    }

    #[inline]
    pub fn mb(&self, x: usize, y: usize) -> ColMb {
        self.mbs.get(y * self.mb_w + x).copied().unwrap_or_default()
    }
}

/// Which part of a frame a picture or a reference covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Structure {
    Frame,
    Top,
    Bottom,
}

impl Structure {
    pub fn field(parity: u8) -> Self {
        if parity == 0 { Structure::Top } else { Structure::Bottom }
    }
    /// 0 = top, 1 = bottom; None for a frame.
    pub fn parity(self) -> Option<u8> {
        match self {
            Structure::Frame => None,
            Structure::Top => Some(0),
            Structure::Bottom => Some(1),
        }
    }
    pub fn is_field(self) -> bool {
        self != Structure::Frame
    }
}

/// A decoded (or in-progress) frame as stored in the DPB and referenced by later pictures.
///
/// Rows are published in order by the thread decoding the picture; readers block (per row) until the
/// data they need is available, which lets several pictures decode concurrently.
///
/// Frames of interlaced streams (`frame_mbs_only_flag` 0) are stored as two fields: each row is a
/// field macroblock row (16 lines of one field), top field rows first. Their motion data for
/// direct prediction lives in [`ColPic`]s published when a picture is complete.
pub struct Frame {
    pub id: u32,
    pub poc: i32,
    pub width: usize,
    pub height: usize,
    pub cwidth: usize,
    pub cheight: usize,
    pub mb_w: usize,
    pub interlaced: bool,
    rows: Box<[OnceLock<FrameRow>]>,
    /// Interlaced only: motion of the coded frame ([0]) or of the top / bottom field ([0] / [1]).
    col: [OnceLock<Arc<ColPic>>; 2],
}

impl Frame {
    pub fn new(id: u32, poc: i32, mb_w: usize, mb_h: usize) -> Self {
        Frame {
            id,
            poc,
            width: mb_w * 16,
            height: mb_h * 16,
            cwidth: mb_w * 8,
            cheight: mb_h * 8,
            mb_w,
            interlaced: false,
            rows: (0..mb_h).map(|_| OnceLock::new()).collect(),
            col: [OnceLock::new(), OnceLock::new()],
        }
    }

    /// A frame of an interlaced stream (`mb_h` frame macroblock rows, even).
    pub fn new_interlaced(id: u32, poc: i32, mb_w: usize, mb_h: usize) -> Self {
        let mb_h = mb_h & !1;
        Frame { interlaced: true, ..Frame::new(id, poc, mb_w, mb_h) }
    }

    /// Frame macroblock rows.
    pub fn mb_h(&self) -> usize {
        self.rows.len()
    }

    /// Macroblock rows of one field (interlaced frames).
    pub fn field_mb_h(&self) -> usize {
        self.rows.len() / 2
    }

    /// Publish MB row `r` (ignored if already published).
    pub fn publish(&self, r: usize, row: FrameRow) {
        if let Some(slot) = self.rows.get(r) {
            let _ = slot.set(row);
        }
    }

    /// Publish field MB row `r` of field `parity` (interlaced frames).
    pub fn publish_field(&self, parity: u8, r: usize, row: FrameRow) {
        self.publish(parity as usize * self.field_mb_h() + r, row);
    }

    pub fn is_published(&self, r: usize) -> bool {
        self.rows.get(r).is_some_and(|s| s.get().is_some())
    }

    pub fn is_field_published(&self, parity: u8, r: usize) -> bool {
        self.is_published(parity as usize * self.field_mb_h() + r)
    }

    /// Every row has been published (rows are published in order within each field).
    pub fn is_complete(&self) -> bool {
        if self.interlaced {
            let n = self.field_mb_h();
            n == 0 || (self.is_field_published(0, n - 1) && self.is_field_published(1, n - 1))
        } else {
            self.rows.is_empty() || self.is_published(self.rows.len() - 1)
        }
    }

    /// The part `s` (frame or one field) has been published completely.
    #[cfg_attr(not(feature = "threads"), allow(dead_code))]
    pub fn is_part_complete(&self, s: Structure) -> bool {
        match s.parity() {
            Some(p) if self.interlaced => {
                let n = self.field_mb_h();
                n == 0 || self.is_field_published(p, n - 1)
            }
            _ => self.is_complete(),
        }
    }

    /// MB row `r`, waiting until it has been published.
    #[inline]
    pub fn row(&self, r: usize) -> &FrameRow {
        #[cfg(feature = "threads")]
        {
            self.rows[r].wait()
        }
        #[cfg(not(feature = "threads"))]
        {
            self.rows[r].get().expect("reference row decoded before use")
        }
    }

    /// Field MB row `r` of field `parity` (interlaced frames), waiting until published.
    #[inline]
    pub fn field_row(&self, parity: u8, r: usize) -> &FrameRow {
        self.row(parity as usize * self.field_mb_h() + r)
    }

    /// Wait until the whole frame is available.
    pub fn wait_complete(&self) {
        for r in 0..self.rows.len() {
            self.row(r);
        }
    }

    /// Wait until part `s` is available.
    pub fn wait_part(&self, s: Structure) {
        match s.parity() {
            Some(p) if self.interlaced => {
                for r in 0..self.field_mb_h() {
                    self.field_row(p, r);
                }
            }
            _ => self.wait_complete(),
        }
    }

    /// Publish the motion data of a coded field (`idx` = parity); a coded frame goes to both.
    pub fn publish_col(&self, idx: usize, col: Arc<ColPic>) {
        if let Some(slot) = self.col.get(idx) {
            let _ = slot.set(col);
        }
    }

    /// Motion data of the coded picture covering field `parity` (a coded frame is published for
    /// both parities). Waits until available.
    pub fn col(&self, parity: u8) -> Option<&Arc<ColPic>> {
        self.wait_col(parity as usize & 1)
    }

    fn wait_col(&self, idx: usize) -> Option<&Arc<ColPic>> {
        let slot = self.col.get(idx)?;
        #[cfg(feature = "threads")]
        {
            Some(slot.wait())
        }
        #[cfg(not(feature = "threads"))]
        {
            slot.get()
        }
    }

    /// Build a published row from a contiguous plane set (rows of MB row `r`) and motion data.
    pub fn make_row(planes: &Planes, r: usize, mbs: &[MbState], ref_ids: &dyn Fn(&MbState, usize, i8) -> u32) -> FrameRow {
        let w = planes.width;
        let cw = planes.cwidth;
        let mb_w = w / 16;
        let y = planes.y[r * 16 * w..(r + 1) * 16 * w].into();
        let cb = planes.cb[r * 8 * cw..(r + 1) * 8 * cw].into();
        let cr = planes.cr[r * 8 * cw..(r + 1) * 8 * cw].into();
        let row_mbs = &mbs[r * mb_w..(r + 1) * mb_w];
        let mut mv = [vec![[0i16; 2]; mb_w * 16], vec![[0i16; 2]; mb_w * 16]];
        let mut ref_idx = [vec![-1i8; mb_w * 4], vec![-1i8; mb_w * 4]];
        let mut ref_id = [vec![u32::MAX; mb_w * 4], vec![u32::MAX; mb_w * 4]];
        let mut intra = vec![true; mb_w];
        for (x, st) in row_mbs.iter().enumerate() {
            if st.slice_num == u32::MAX {
                continue;
            }
            intra[x] = st.kind.is_intra();
            for l in 0..2 {
                mv[l][x * 16..x * 16 + 16].copy_from_slice(&st.mv[l]);
                for b in 0..4 {
                    let ri = st.ref_idx[l][b];
                    ref_idx[l][x * 4 + b] = ri;
                    if ri >= 0 {
                        ref_id[l][x * 4 + b] = ref_ids(st, l, ri);
                    }
                }
            }
        }
        let [mv0, mv1] = mv;
        let [ri0, ri1] = ref_idx;
        let [id0, id1] = ref_id;
        FrameRow { y, cb, cr, mv: [mv0.into(), mv1.into()], ref_idx: [ri0.into(), ri1.into()], ref_id: [id0.into(), id1.into()], intra: intra.into() }
    }

    /// A sample-only row (interlaced storage): 16 luma lines starting at line `y0` and 8 chroma
    /// lines starting at `cy0`, taking every `step`-th line of `planes`.
    pub fn sample_row(planes: &Planes, y0: usize, cy0: usize, step: usize) -> FrameRow {
        let (w, cw) = (planes.width, planes.cwidth);
        let mut y = Vec::with_capacity(16 * w);
        for i in 0..16 {
            let o = (y0 + i * step) * w;
            y.extend_from_slice(planes.y.get(o..o + w).unwrap_or(&[]));
        }
        y.resize(16 * w, 128);
        let mut cb = Vec::with_capacity(8 * cw);
        let mut cr = Vec::with_capacity(8 * cw);
        for i in 0..8 {
            let o = (cy0 + i * step) * cw;
            cb.extend_from_slice(planes.cb.get(o..o + cw).unwrap_or(&[]));
            cr.extend_from_slice(planes.cr.get(o..o + cw).unwrap_or(&[]));
        }
        cb.resize(8 * cw, 128);
        cr.resize(8 * cw, 128);
        FrameRow {
            y: y.into(),
            cb: cb.into(),
            cr: cr.into(),
            mv: [Box::new([]), Box::new([])],
            ref_idx: [Box::new([]), Box::new([])],
            ref_id: [Box::new([]), Box::new([])],
            intra: Box::new([]),
        }
    }

    /// Publish field rows `from_row..to_row` of field `parity` of an interlaced frame from field
    /// planes (`planes` holds that field only).
    pub fn publish_field_planes(&self, parity: u8, planes: &Planes, from_row: usize, to_row: usize) {
        for r in from_row..to_row.min(self.field_mb_h()) {
            self.publish_field(parity, r, Frame::sample_row(planes, r * 16, r * 8, 1));
        }
    }

    /// Publish field rows `from_pair..to_pair` of both fields from frame planes (interlaced frames).
    pub fn publish_frame_planes(&self, planes: &Planes, from_pair: usize, to_pair: usize) {
        for r in from_pair..to_pair.min(self.field_mb_h()) {
            for p in 0..2u8 {
                self.publish_field(p, r, Frame::sample_row(planes, r * 32 + p as usize, r * 16 + p as usize, 2));
            }
        }
    }

    /// A frame with every row published from `planes` and no motion (intra), e.g. "non-existing"
    /// frames inferred for frame_num gaps.
    pub fn from_planes(id: u32, poc: i32, planes: &Planes) -> Self {
        let (mb_w, mb_h) = (planes.width / 16, planes.height / 16);
        let f = Frame::new(id, poc, mb_w, mb_h);
        let blank = vec![MbState::default(); mb_w * mb_h];
        for r in 0..mb_h {
            f.publish(r, Frame::make_row(planes, r, &blank, &|_, _, _| u32::MAX));
        }
        f
    }

    /// Interlaced counterpart of [`Frame::from_planes`] (frame planes, both fields).
    pub fn from_planes_interlaced(id: u32, poc: i32, planes: &Planes) -> Self {
        let (mb_w, mb_h) = (planes.width / 16, planes.height / 16);
        let f = Frame::new_interlaced(id, poc, mb_w, mb_h);
        f.publish_frame_planes(planes, 0, f.field_mb_h());
        let col = Arc::new(ColPic::intra(ColKind::Frm, mb_w, f.mb_h()));
        f.publish_col(0, col.clone());
        f.publish_col(1, col);
        f
    }

    /// Fill field `missing` of an interlaced frame whose other field was decoded alone (a
    /// non-paired field): the missing field repeats the lines of the decoded one.
    pub fn fill_missing_field(&self, missing: u8) {
        let have = missing ^ 1;
        for r in 0..self.field_mb_h() {
            if self.is_field_published(missing, r) {
                continue;
            }
            let src = self.field_row(have, r);
            let row = FrameRow {
                y: src.y.clone(),
                cb: src.cb.clone(),
                cr: src.cr.clone(),
                mv: src.mv.clone(),
                ref_idx: src.ref_idx.clone(),
                ref_id: src.ref_id.clone(),
                intra: src.intra.clone(),
            };
            self.publish_field(missing, r, row);
        }
        let n = self.field_mb_h();
        self.publish_col(missing as usize, Arc::new(ColPic::intra(ColKind::Fld, self.mb_w, n)));
    }

    /// Copy the w x h luma window at (x0, y0) (clamped to the picture) into `out` (stride `os`).
    #[inline]
    pub fn luma_window(&self, x0: i32, y0: i32, w: usize, h: usize, out: &mut [u8], os: usize) {
        let (width, height) = (self.width, self.height);
        if self.interlaced {
            for r in 0..h {
                let yy = (y0 + r as i32).clamp(0, height as i32 - 1) as usize;
                copy_line::<24>(self.luma_line(yy), x0, &mut out[r * os..], w);
            }
            return;
        }
        for r in 0..h {
            let yy = (y0 + r as i32).clamp(0, height as i32 - 1) as usize;
            let row = self.row(yy >> 4);
            let line = &row.y[(yy & 15) * width..(yy & 15) * width + width];
            copy_line::<24>(line, x0, &mut out[r * os..], w);
        }
    }

    /// Frame luma line `yy` (any storage).
    #[inline(always)]
    fn luma_line(&self, yy: usize) -> &[u8] {
        let width = self.width;
        if self.interlaced {
            let fy = yy >> 1;
            let row = self.field_row((yy & 1) as u8, fy >> 4);
            &row.y[(fy & 15) * width..(fy & 15) * width + width]
        } else {
            let row = self.row(yy >> 4);
            &row.y[(yy & 15) * width..(yy & 15) * width + width]
        }
    }

    /// Frame chroma line `yy` of component `c` (any storage).
    #[inline(always)]
    fn chroma_line(&self, c: usize, yy: usize) -> &[u8] {
        let cw = self.cwidth;
        let (row, l) = if self.interlaced {
            let fy = yy >> 1;
            (self.field_row((yy & 1) as u8, fy >> 3), fy & 7)
        } else {
            (self.row(yy >> 3), yy & 7)
        };
        let plane = if c == 0 { &row.cb } else { &row.cr };
        &plane[l * cw..l * cw + cw]
    }

    /// Luma window of field `parity` (interlaced frames): (x0, y0) in field coordinates.
    #[inline]
    pub fn field_luma_window(&self, parity: u8, x0: i32, y0: i32, w: usize, h: usize, out: &mut [u8], os: usize) {
        let width = self.width;
        let fh = (self.height / 2) as i32;
        for r in 0..h {
            let yy = (y0 + r as i32).clamp(0, fh - 1) as usize;
            let row = self.field_row(parity, yy >> 4);
            let line = &row.y[(yy & 15) * width..(yy & 15) * width + width];
            copy_line::<24>(line, x0, &mut out[r * os..], w);
        }
    }

    /// Copy the w x h window of chroma component `c` at (x0, y0) (clamped) into `out`.
    #[inline]
    pub fn chroma_window(&self, c: usize, x0: i32, y0: i32, w: usize, h: usize, out: &mut [u8], os: usize) {
        let (cw, ch) = (self.cwidth, self.cheight);
        if self.interlaced {
            for r in 0..h {
                let yy = (y0 + r as i32).clamp(0, ch as i32 - 1) as usize;
                copy_line::<16>(self.chroma_line(c, yy), x0, &mut out[r * os..], w);
            }
            return;
        }
        for r in 0..h {
            let yy = (y0 + r as i32).clamp(0, ch as i32 - 1) as usize;
            let row = self.row(yy >> 3);
            let plane = if c == 0 { &row.cb } else { &row.cr };
            let line = &plane[(yy & 7) * cw..(yy & 7) * cw + cw];
            copy_line::<16>(line, x0, &mut out[r * os..], w);
        }
    }

    /// Chroma window of field `parity` (interlaced frames), field coordinates.
    #[inline]
    pub fn field_chroma_window(&self, parity: u8, c: usize, x0: i32, y0: i32, w: usize, h: usize, out: &mut [u8], os: usize) {
        let cw = self.cwidth;
        let fch = (self.cheight / 2) as i32;
        for r in 0..h {
            let yy = (y0 + r as i32).clamp(0, fch - 1) as usize;
            let row = self.field_row(parity, yy >> 3);
            let plane = if c == 0 { &row.cb } else { &row.cr };
            let line = &plane[(yy & 7) * cw..(yy & 7) * cw + cw];
            copy_line::<16>(line, x0, &mut out[r * os..], w);
        }
    }

    /// Crop and copy the frame into planar buffers (waits for completion).
    pub fn copy_cropped(&self, crop: (usize, usize, usize, usize)) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (cx, cy, cw, ch) = crop;
        let alloc = plane_allocator();
        let mut y = alloc(cw * ch);
        for r in cy..cy + ch {
            y.extend_from_slice(&self.luma_line(r)[cx..cx + cw]);
        }
        let (ccx, ccy, ccw, cch) = (cx / 2, cy / 2, cw.div_ceil(2), ch.div_ceil(2));
        let mut u = alloc(ccw * cch);
        let mut v = alloc(ccw * cch);
        for r in ccy..ccy + cch {
            u.extend_from_slice(&self.chroma_line(0, r)[ccx..ccx + ccw]);
            v.extend_from_slice(&self.chroma_line(1, r)[ccx..ccx + ccw]);
        }
        (y, u, v)
    }
}

static PLANE_ALLOCATOR: std::sync::OnceLock<fn(usize) -> Vec<u8>> = std::sync::OnceLock::new();

/// Where output planes come from: an empty buffer with room for the given number of samples.
/// The host sets it once to recycle the planes of pictures it is done with; the first call wins.
pub fn set_plane_allocator(alloc: fn(usize) -> Vec<u8>) {
    let _ = PLANE_ALLOCATOR.set(alloc);
}

fn plane_allocator() -> fn(usize) -> Vec<u8> {
    PLANE_ALLOCATOR.get().copied().unwrap_or(Vec::with_capacity)
}

/// Copy `w` samples of `line` starting at x0 into `out`; when possible a fixed-size block of `N`
/// bytes is copied instead (cheaper than a variable-length copy; `out` must then have room for `N`).
#[inline(always)]
fn copy_line<const N: usize>(line: &[u8], x0: i32, out: &mut [u8], w: usize) {
    if w <= N
        && x0 >= 0
        && let Some(src) = line.get(x0 as usize..).and_then(|l| l.first_chunk::<N>())
        && let Some(dst) = out.first_chunk_mut::<N>()
    {
        *dst = *src;
    } else {
        copy_clamped(line, x0, &mut out[..w]);
    }
}

/// Copy `out.len()` samples of `line` starting at x0, replicating edge samples outside the line.
#[inline(always)]
fn copy_clamped(line: &[u8], x0: i32, out: &mut [u8]) {
    let w = out.len();
    if x0 >= 0 && x0 as usize + w <= line.len() {
        out.copy_from_slice(&line[x0 as usize..x0 as usize + w]);
    } else {
        let max = line.len() as i32 - 1;
        for (c, o) in out.iter_mut().enumerate() {
            *o = line[(x0 + c as i32).clamp(0, max) as usize];
        }
    }
}

pub type FrameRef = Arc<Frame>;

/// An entry of RefPicList0/1 for the current slice.
#[derive(Clone)]
pub struct RefPic {
    pub frame: FrameRef,
    pub long_term: bool,
    /// Frame or field of `frame`.
    pub structure: Structure,
    /// PicOrderCnt of the frame (Min of its fields) or of the field.
    pub poc: i32,
    /// TopFieldOrderCnt / BottomFieldOrderCnt of the frame.
    pub field_poc: [i32; 2],
}

impl RefPic {
    pub fn poc(&self) -> i32 {
        self.poc
    }

    /// Unique identity of the referenced frame or field.
    pub fn id(&self) -> u32 {
        Self::make_id(self.frame.id, self.structure)
    }

    pub fn make_id(frame_id: u32, s: Structure) -> u32 {
        (frame_id << 2)
            | match s {
                Structure::Frame => 0,
                Structure::Top => 1,
                Structure::Bottom => 2,
            }
    }

    /// Field `parity` of this frame reference (MBAFF field macroblocks).
    pub fn field_of(&self, parity: u8) -> RefPic {
        RefPic {
            frame: self.frame.clone(),
            long_term: self.long_term,
            structure: Structure::field(parity),
            poc: self.field_poc[parity as usize & 1],
            field_poc: self.field_poc,
        }
    }

    /// Luma window (see [`Frame::luma_window`]) of the frame or field.
    #[inline]
    pub fn luma_window(&self, x0: i32, y0: i32, w: usize, h: usize, out: &mut [u8], os: usize) {
        match self.structure.parity() {
            Some(p) if self.frame.interlaced => self.frame.field_luma_window(p, x0, y0, w, h, out, os),
            _ => self.frame.luma_window(x0, y0, w, h, out, os),
        }
    }

    #[inline]
    pub fn chroma_window(&self, c: usize, x0: i32, y0: i32, w: usize, h: usize, out: &mut [u8], os: usize) {
        match self.structure.parity() {
            Some(p) if self.frame.interlaced => self.frame.field_chroma_window(p, c, x0, y0, w, h, out, os),
            _ => self.frame.chroma_window(c, x0, y0, w, h, out, os),
        }
    }
}
