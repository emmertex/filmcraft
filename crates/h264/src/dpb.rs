//! Decoded picture buffer: reference marking (8.2.5), reference list construction (8.2.4) and output
//! ordering / bumping (C.4).
//!
//! The buffer only does bookkeeping on picture headers; what a picture is (`F`: the software
//! decoder's [`crate::picture::FrameRef`], a hardware decoder's surface) is up to the decoder, so
//! hardware decoders driven by the same headers (VA-API) share this logic and output order.
//!
//! Each entry is a frame buffer: a decoded frame, a complementary field pair or a single field.
//! Reference marking is kept per field (`field_mark`); `mark` summarises it for frame decoding.

use crate::error::{Result, ensure};
use crate::slice::{Mmco, SliceHeader, SliceType};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefMark {
    Unused,
    Short,
    Long,
}

/// Output-related metadata attached to a decoded frame.
#[derive(Clone, Debug)]
pub struct OutputMeta {
    pub pts: i64,
    pub key: bool,
    pub crop: (u32, u32, u32, u32),
    pub full_range: bool,
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub sar: (u16, u16),
    /// Decoded without deblocking (draft mode, non-reference picture).
    pub draft: bool,
}

pub struct DpbEntry<F> {
    pub frame: F,
    /// PicOrderCnt of the frame: Min of the decoded fields' order counts.
    pub poc: i32,
    pub frame_num: u32,
    pub long_term_frame_idx: u32,
    /// Frame-level marking: the common marking of the decoded fields (any reference marking when
    /// they differ).
    pub mark: RefMark,
    pub needed_for_output: bool,
    pub non_existing: bool,
    pub meta: Arc<OutputMeta>,
    /// TopFieldOrderCnt / BottomFieldOrderCnt.
    pub field_poc: [i32; 2],
    /// Reference marking of the top / bottom field.
    pub field_mark: [RefMark; 2],
    /// Decoded fields: bit 0 top, bit 1 bottom (3 for a frame or a complementary field pair).
    pub fields: u8,
    /// A first field whose second field may still follow (not output until resolved).
    pub awaiting_second: bool,
}

impl<F> DpbEntry<F> {
    fn set_mark(&mut self, m: RefMark) {
        self.mark = m;
        self.field_mark = [m; 2];
    }

    fn set_field_mark(&mut self, parity: u8, m: RefMark) {
        self.field_mark[parity as usize & 1] = m;
        self.sync_mark();
    }

    /// Recompute `mark` from the field marks of the decoded fields.
    fn sync_mark(&mut self) {
        let present: Vec<RefMark> = (0..2).filter(|p| self.fields & (1 << p) != 0).map(|p| self.field_mark[p]).collect();
        self.mark = if present.iter().all(|m| *m == RefMark::Long) && !present.is_empty() {
            RefMark::Long
        } else if present.contains(&RefMark::Short) {
            RefMark::Short
        } else if present.contains(&RefMark::Long) {
            RefMark::Long
        } else {
            RefMark::Unused
        };
    }

    fn is_ref(&self) -> bool {
        self.field_mark.iter().any(|m| *m != RefMark::Unused)
    }

    /// Both fields marked `m` (a reference frame / complementary reference field pair).
    fn frame_marked(&self, m: RefMark) -> bool {
        self.fields == 3 && self.field_mark == [m; 2]
    }

    fn field_marked(&self, parity: u8, m: RefMark) -> bool {
        self.fields & (1 << parity) != 0 && self.field_mark[parity as usize & 1] == m
    }
}

pub struct Dpb<F> {
    pub entries: Vec<DpbEntry<F>>,
    /// None = "no long-term frame indices".
    pub max_long_term_frame_idx: Option<u32>,
    pub capacity: usize,
    pub max_reorder: usize,
}

/// A picture leaving the DPB for output.
pub struct Output<F> {
    pub frame: F,
    pub poc: i32,
    pub meta: Arc<OutputMeta>,
}

/// A reference list element under construction: a DPB entry and the field (None = the frame).
type Item<'a, F> = (&'a DpbEntry<F>, Option<u8>);

impl<F: Clone> Dpb<F> {
    pub fn new() -> Self {
        Dpb { entries: Vec::new(), max_long_term_frame_idx: None, capacity: 16, max_reorder: 16 }
    }

    fn frame_num_wrap(e: &DpbEntry<F>, cur_frame_num: u32, max_frame_num: u32) -> i32 {
        if e.frame_num > cur_frame_num { e.frame_num as i32 - max_frame_num as i32 } else { e.frame_num as i32 }
    }

    fn num_refs(&self) -> usize {
        self.entries.iter().filter(|e| e.is_ref()).count()
    }

    /// Remove entries not needed for output and not used for reference.
    fn prune(&mut self) {
        self.entries.retain(|e| e.needed_for_output || e.is_ref() || e.awaiting_second);
    }

    /// Output the picture with the smallest POC (C.4.5.3 "bumping"). Returns false if none waits
    /// (or the next one is a first field still waiting for its second field).
    fn bump(&mut self, out: &mut Vec<Output<F>>) -> bool {
        let Some(i) = self.entries.iter().enumerate().filter(|(_, e)| e.needed_for_output).min_by_key(|(_, e)| e.poc).map(|(i, _)| i) else {
            return false;
        };
        let e = &mut self.entries[i];
        if e.awaiting_second {
            return false;
        }
        e.needed_for_output = false;
        out.push(Output { frame: e.frame.clone(), poc: e.poc, meta: e.meta.clone() });
        self.prune();
        true
    }

    /// Output everything in POC order and clear non-reference entries.
    pub fn flush(&mut self, out: &mut Vec<Output<F>>) {
        self.close_pending();
        while self.bump(out) {}
        self.prune();
    }

    /// IDR: mark all references unused; output (or drop) prior pictures.
    pub fn idr(&mut self, no_output_of_prior_pics: bool, out: &mut Vec<Output<F>>) {
        self.close_pending();
        for e in &mut self.entries {
            e.set_mark(RefMark::Unused);
        }
        if no_output_of_prior_pics {
            self.entries.clear();
        } else {
            self.flush(out);
        }
        self.max_long_term_frame_idx = None;
    }

    /// The first field waiting for its second field, if any: (frame, parity, frame_num, reference).
    pub fn pending_first_field(&self) -> Option<(&F, u8, u32, bool)> {
        self.entries.iter().find(|e| e.awaiting_second).map(|e| (&e.frame, if e.fields & 1 != 0 { 0 } else { 1 }, e.frame_num, e.is_ref()))
    }

    /// The pending first field (if any) stays a non-paired field.
    pub fn close_pending(&mut self) {
        for e in &mut self.entries {
            e.awaiting_second = false;
        }
    }

    /// Sliding window marking (8.2.5.3).
    fn sliding_window(&mut self, cur_frame_num: u32, max_frame_num: u32, max_num_ref_frames: usize) {
        let limit = max_num_ref_frames.max(1);
        while self.num_refs() >= limit {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.field_mark.contains(&RefMark::Short))
                .min_by_key(|(_, e)| Self::frame_num_wrap(e, cur_frame_num, max_frame_num))
                .map(|(i, _)| i);
            match victim {
                Some(i) => {
                    let e = &mut self.entries[i];
                    for p in 0..2 {
                        if e.field_mark[p] == RefMark::Short {
                            e.field_mark[p] = RefMark::Unused;
                        }
                    }
                    e.sync_mark();
                }
                None => break,
            }
        }
        self.prune();
    }

    /// Insert "non-existing" frames for a frame_num gap (8.2.5.2). `make_frame` gives each one's
    /// picture and POC.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_frame_num_gap(
        &mut self,
        prev_ref_frame_num: u32,
        frame_num: u32,
        max_frame_num: u32,
        max_num_ref_frames: usize,
        make_frame: &mut dyn FnMut(u32) -> (F, i32, Arc<OutputMeta>),
        poc_state: &mut dyn FnMut(u32),
    ) {
        self.close_pending();
        let mut unused = (prev_ref_frame_num + 1) % max_frame_num;
        let mut guard = 0;
        while unused != frame_num && guard < max_frame_num {
            self.sliding_window(unused, max_frame_num, max_num_ref_frames);
            let (frame, poc, meta) = make_frame(unused);
            poc_state(unused);
            self.entries.push(DpbEntry {
                poc,
                frame,
                frame_num: unused,
                long_term_frame_idx: 0,
                mark: RefMark::Short,
                needed_for_output: false,
                non_existing: true,
                meta,
                field_poc: [poc, poc],
                field_mark: [RefMark::Short; 2],
                fields: 3,
                awaiting_second: false,
            });
            unused = (unused + 1) % max_frame_num;
            guard += 1;
        }
    }

    /// Initial reference picture lists (8.2.4.2) + modification (8.2.4.3) for a frame slice, each
    /// entry made by `to_ref` from the DPB entry it refers to.
    pub fn build_ref_lists<R>(&self, sh: &SliceHeader, cur_poc: i32, max_frame_num: u32, to_ref: impl Fn(&DpbEntry<F>) -> R) -> Result<[Vec<R>; 2]> {
        let lists = self.build_lists(sh, cur_poc, max_frame_num, None)?;
        Ok(lists.map(|l| l.into_iter().map(|(e, _)| to_ref(e)).collect()))
    }

    /// Reference lists of a field slice (field `parity` is being decoded); `to_ref` receives the
    /// DPB entry and the referenced field's parity.
    pub fn build_field_ref_lists<R>(
        &self,
        sh: &SliceHeader,
        cur_poc: i32,
        max_frame_num: u32,
        parity: u8,
        to_ref: impl Fn(&DpbEntry<F>, u8) -> R,
    ) -> Result<[Vec<R>; 2]> {
        let lists = self.build_lists(sh, cur_poc, max_frame_num, Some(parity))?;
        Ok(lists.map(|l| l.into_iter().map(|(e, p)| to_ref(e, p.unwrap_or(parity))).collect()))
    }

    /// Alternate fields of `frames` starting with parity `same` (8.2.4.2.5); only fields marked `m`.
    fn alternate<'a>(frames: &[&'a DpbEntry<F>], same: u8, m: RefMark) -> Vec<Item<'a, F>> {
        let mut out = Vec::new();
        let mut idx = [0usize; 2];
        let mut parity = same;
        let next = |idx: &mut usize, p: u8| -> Option<&'a DpbEntry<F>> {
            while *idx < frames.len() {
                let e = frames[*idx];
                *idx += 1;
                if e.field_marked(p, m) {
                    return Some(e);
                }
            }
            None
        };
        loop {
            match next(&mut idx[parity as usize], parity) {
                Some(e) => out.push((e, Some(parity))),
                None => {
                    // no more fields of this parity: append the rest of the other one in order
                    let other = parity ^ 1;
                    while let Some(e) = next(&mut idx[other as usize], other) {
                        out.push((e, Some(other)));
                    }
                    break;
                }
            }
            parity ^= 1;
        }
        out
    }

    fn build_lists<'a>(&'a self, sh: &SliceHeader, cur_poc: i32, max_frame_num: u32, field: Option<u8>) -> Result<[Vec<Item<'a, F>>; 2]> {
        let mut lists: [Vec<Item<'a, F>>; 2] = [Vec::new(), Vec::new()];
        if sh.slice_type.is_intra() {
            return Ok(lists);
        }
        let cur_fn = sh.frame_num;
        let fnw = |e: &DpbEntry<F>| Self::frame_num_wrap(e, cur_fn, max_frame_num);
        let mut entry_lists: [Vec<Item<'a, F>>; 2] = [Vec::new(), Vec::new()];
        match field {
            None => {
                let short: Vec<&DpbEntry<F>> = self.entries.iter().filter(|e| e.frame_marked(RefMark::Short)).collect();
                let mut long: Vec<&DpbEntry<F>> = self.entries.iter().filter(|e| e.frame_marked(RefMark::Long)).collect();
                long.sort_by_key(|e| e.long_term_frame_idx);
                if sh.slice_type != SliceType::B {
                    let mut s = short.clone();
                    s.sort_by_key(|e| std::cmp::Reverse(fnw(e)));
                    entry_lists[0] = s.into_iter().chain(long.iter().copied()).map(|e| (e, None)).collect();
                } else {
                    let mut before: Vec<&DpbEntry<F>> = short.iter().copied().filter(|e| e.poc < cur_poc).collect();
                    let mut after: Vec<&DpbEntry<F>> = short.iter().copied().filter(|e| e.poc > cur_poc).collect();
                    before.sort_by_key(|e| std::cmp::Reverse(e.poc));
                    after.sort_by_key(|e| e.poc);
                    entry_lists[0] = before.iter().chain(after.iter()).chain(long.iter()).map(|e| (*e, None)).collect();
                    entry_lists[1] = after.iter().chain(before.iter()).chain(long.iter()).map(|e| (*e, None)).collect();
                }
            }
            Some(parity) => {
                let short: Vec<&DpbEntry<F>> = self.entries.iter().filter(|e| e.field_mark.contains(&RefMark::Short)).collect();
                let mut long: Vec<&DpbEntry<F>> = self.entries.iter().filter(|e| e.field_mark.contains(&RefMark::Long)).collect();
                long.sort_by_key(|e| e.long_term_frame_idx);
                let long_fields = Self::alternate(&long, parity, RefMark::Long);
                if sh.slice_type != SliceType::B {
                    let mut s = short.clone();
                    s.sort_by_key(|e| std::cmp::Reverse(fnw(e)));
                    entry_lists[0] = Self::alternate(&s, parity, RefMark::Short).into_iter().chain(long_fields).collect();
                } else {
                    // POC of a reference entry: of its fields marked as short-term reference
                    let epoc =
                        |e: &DpbEntry<F>| (0..2u8).filter(|&p| e.field_marked(p, RefMark::Short)).map(|p| e.field_poc[p as usize]).min().unwrap_or(e.poc);
                    let mut before: Vec<&DpbEntry<F>> = short.iter().copied().filter(|e| epoc(e) <= cur_poc).collect();
                    let mut after: Vec<&DpbEntry<F>> = short.iter().copied().filter(|e| epoc(e) > cur_poc).collect();
                    before.sort_by_key(|e| std::cmp::Reverse(epoc(e)));
                    after.sort_by_key(|e| epoc(e));
                    let l0: Vec<&DpbEntry<F>> = before.iter().chain(after.iter()).copied().collect();
                    let l1: Vec<&DpbEntry<F>> = after.iter().chain(before.iter()).copied().collect();
                    entry_lists[0] = Self::alternate(&l0, parity, RefMark::Short).into_iter().chain(long_fields.iter().copied()).collect();
                    entry_lists[1] = Self::alternate(&l1, parity, RefMark::Short).into_iter().chain(long_fields).collect();
                }
            }
        }
        if sh.slice_type == SliceType::B && entry_lists[1].len() > 1 && entry_lists[0].len() == entry_lists[1].len() {
            let same = entry_lists[0].iter().zip(entry_lists[1].iter()).all(|(a, b)| std::ptr::eq(a.0, b.0) && a.1 == b.1);
            if same {
                entry_lists[1].swap(0, 1);
            }
        }
        // picture numbers (8.2.4.1)
        let same_parity = |p: Option<u8>| match (field, p) {
            (Some(c), Some(p)) => (c == p) as i32,
            _ => 0,
        };
        let pic_num = |it: &Item<F>| if field.is_some() { 2 * fnw(it.0) + same_parity(it.1) } else { fnw(it.0) };
        let long_pic_num =
            |it: &Item<F>| if field.is_some() { 2 * it.0.long_term_frame_idx as i32 + same_parity(it.1) } else { it.0.long_term_frame_idx as i32 };
        let is_marked = |it: &Item<F>, m: RefMark| match it.1 {
            Some(p) => it.0.field_marked(p, m),
            None => it.0.frame_marked(m),
        };
        // all candidate fields / frames for modification targets
        let candidates: Vec<Item<'a, F>> = match field {
            None => self.entries.iter().map(|e| (e, None)).collect(),
            Some(_) => self.entries.iter().flat_map(|e| [(e, Some(0u8)), (e, Some(1u8))]).collect(),
        };
        let nlists = if sh.slice_type == SliceType::B { 2 } else { 1 };
        for l in 0..nlists {
            let n = sh.num_ref_idx_active[l] as usize;
            let mut list: Vec<Option<Item<'a, F>>> = entry_lists[l].iter().copied().map(Some).collect();
            list.truncate(n);
            list.resize(n, None);
            if !sh.ref_pic_list_mod[l].is_empty() {
                let (max_pic_num, curr_pic_num) = match field {
                    None => (max_frame_num as i32, cur_fn as i32),
                    Some(_) => (2 * max_frame_num as i32, 2 * cur_fn as i32 + 1),
                };
                let mut pred = curr_pic_num;
                let mut ref_idx = 0usize;
                for m in &sh.ref_pic_list_mod[l] {
                    ensure!(ref_idx < n, "too many ref_pic_list_modification entries");
                    list.push(None); // temporarily one longer
                    let (target, is_long): (Option<Item<'a, F>>, bool) = match m.idc {
                        0 | 1 => {
                            let d = m.value as i32 + 1;
                            let no_wrap = if m.idc == 0 {
                                if pred - d < 0 { pred - d + max_pic_num } else { pred - d }
                            } else if pred + d >= max_pic_num {
                                pred + d - max_pic_num
                            } else {
                                pred + d
                            };
                            pred = no_wrap;
                            let pn = if no_wrap > curr_pic_num { no_wrap - max_pic_num } else { no_wrap };
                            (candidates.iter().copied().find(|it| is_marked(it, RefMark::Short) && pic_num(it) == pn), false)
                        }
                        2 => (candidates.iter().copied().find(|it| is_marked(it, RefMark::Long) && long_pic_num(it) == m.value as i32), true),
                        _ => continue,
                    };
                    let Some(target) = target else {
                        // Missing picture: keep the list unchanged (error concealment).
                        list.pop();
                        ref_idx += 1;
                        continue;
                    };
                    for c in (ref_idx + 1..=n).rev() {
                        list[c] = list[c - 1];
                    }
                    list[ref_idx] = Some(target);
                    ref_idx += 1;
                    let mut nidx = ref_idx;
                    for c in ref_idx..=n {
                        let keep = match list[c] {
                            None => true,
                            Some(it) => {
                                let same = std::ptr::eq(it.0, target.0) && it.1 == target.1;
                                if is_long { !(same && is_marked(&it, RefMark::Long)) } else { !(same && is_marked(&it, RefMark::Short)) }
                            }
                        };
                        if keep {
                            list[nidx] = list[c];
                            nidx += 1;
                        }
                    }
                    list.truncate(n);
                }
            }
            // Fill missing entries (should not happen in conforming streams) with the first valid entry.
            let fallback = list.iter().flatten().next().copied().or_else(|| self.entries.last().map(|e| (e, field)));
            lists[l] = list.iter().filter_map(|e| e.or(fallback)).collect();
        }
        Ok(lists)
    }

    /// Adaptive memory control (8.2.5.4). `field`: the parity of the current field (None for a
    /// frame); `cur` the index of the current frame's entry when it is a second field. Returns
    /// true if MMCO 5 was executed.
    fn apply_mmcos(
        &mut self,
        mmcos: &[Mmco],
        cur_frame_num: u32,
        max_frame_num: u32,
        field: Option<u8>,
        cur: Option<usize>,
        cur_long: &mut Option<u32>,
    ) -> bool {
        let mut had5 = false;
        let curr_pic_num = match field {
            None => cur_frame_num as i32,
            Some(_) => 2 * cur_frame_num as i32 + 1,
        };
        // picture number / long-term picture number of field `p` of entry `e` (frame: p ignored)
        let pic_num = |e: &DpbEntry<F>, p: u8| {
            let w = Self::frame_num_wrap(e, cur_frame_num, max_frame_num);
            match field {
                None => w,
                Some(c) => 2 * w + (c == p) as i32,
            }
        };
        let long_num = |e: &DpbEntry<F>, p: u8| match field {
            None => e.long_term_frame_idx as i32,
            Some(c) => 2 * e.long_term_frame_idx as i32 + (c == p) as i32,
        };
        for m in mmcos {
            match m.op {
                1 => {
                    let pn = curr_pic_num - (m.difference_of_pic_nums_minus1 as i32 + 1);
                    for e in &mut self.entries {
                        match field {
                            None => {
                                if e.frame_marked(RefMark::Short) && pic_num(e, 0) == pn {
                                    e.set_mark(RefMark::Unused);
                                }
                            }
                            Some(_) => {
                                for p in 0..2u8 {
                                    if e.field_marked(p, RefMark::Short) && pic_num(e, p) == pn {
                                        e.set_field_mark(p, RefMark::Unused);
                                    }
                                }
                            }
                        }
                    }
                }
                2 => {
                    for e in &mut self.entries {
                        match field {
                            None => {
                                if e.frame_marked(RefMark::Long) && long_num(e, 0) == m.long_term_pic_num as i32 {
                                    e.set_mark(RefMark::Unused);
                                }
                            }
                            Some(_) => {
                                for p in 0..2u8 {
                                    if e.field_marked(p, RefMark::Long) && long_num(e, p) == m.long_term_pic_num as i32 {
                                        e.set_field_mark(p, RefMark::Unused);
                                    }
                                }
                            }
                        }
                    }
                }
                3 => {
                    let pn = curr_pic_num - (m.difference_of_pic_nums_minus1 as i32 + 1);
                    let idx = m.long_term_frame_idx;
                    // the frame / field picNumX
                    let target = self.entries.iter().enumerate().find_map(|(i, e)| match field {
                        None => (e.frame_marked(RefMark::Short) && pic_num(e, 0) == pn).then_some((i, None)),
                        Some(_) => (0..2u8).find(|&p| e.field_marked(p, RefMark::Short) && pic_num(e, p) == pn).map(|p| (i, Some(p))),
                    });
                    for (i, e) in self.entries.iter_mut().enumerate() {
                        if e.long_term_frame_idx != idx || !e.field_mark.contains(&RefMark::Long) {
                            continue;
                        }
                        // a long-term field of the same frame as picNumX keeps its index (pair)
                        if field.is_some() && target.is_some_and(|(t, _)| t == i) {
                            continue;
                        }
                        for p in 0..2 {
                            if e.field_mark[p] == RefMark::Long {
                                e.field_mark[p] = RefMark::Unused;
                            }
                        }
                        e.sync_mark();
                    }
                    if let Some((i, p)) = target {
                        let e = &mut self.entries[i];
                        match p {
                            None => e.set_mark(RefMark::Long),
                            Some(p) => e.set_field_mark(p, RefMark::Long),
                        }
                        e.long_term_frame_idx = idx;
                    }
                }
                4 => {
                    let max = m.max_long_term_frame_idx_plus1;
                    for e in &mut self.entries {
                        if max == 0 || e.long_term_frame_idx > max - 1 {
                            for p in 0..2 {
                                if e.field_mark[p] == RefMark::Long {
                                    e.field_mark[p] = RefMark::Unused;
                                }
                            }
                            e.sync_mark();
                        }
                    }
                    self.max_long_term_frame_idx = if max == 0 { None } else { Some(max - 1) };
                }
                5 => {
                    for e in &mut self.entries {
                        e.set_mark(RefMark::Unused);
                    }
                    self.max_long_term_frame_idx = None;
                    had5 = true;
                }
                6 => {
                    for (i, e) in self.entries.iter_mut().enumerate() {
                        // the first field of the current frame keeps the index it shares
                        if Some(i) == cur {
                            continue;
                        }
                        if e.long_term_frame_idx == m.long_term_frame_idx && e.field_mark.contains(&RefMark::Long) {
                            for p in 0..2 {
                                if e.field_mark[p] == RefMark::Long {
                                    e.field_mark[p] = RefMark::Unused;
                                }
                            }
                            e.sync_mark();
                        }
                    }
                    *cur_long = Some(m.long_term_frame_idx);
                }
                _ => {}
            }
        }
        self.prune();
        had5
    }

    /// Reference marking for the just-decoded picture and insertion into the DPB with output bumping.
    /// Returns pictures ready for output.
    #[allow(clippy::too_many_arguments)]
    pub fn store_picture(
        &mut self,
        sh: &SliceHeader,
        frame: F,
        poc: i32,
        max_frame_num: u32,
        max_num_ref_frames: usize,
        meta: Arc<OutputMeta>,
        out: &mut Vec<Output<F>>,
    ) {
        self.store_frame(sh, frame, [poc, poc], max_frame_num, max_num_ref_frames, meta, out);
    }

    /// [`Dpb::store_picture`] for a frame with distinct field order counts (interlaced streams).
    #[allow(clippy::too_many_arguments)]
    pub fn store_frame(
        &mut self,
        sh: &SliceHeader,
        frame: F,
        field_poc: [i32; 2],
        max_frame_num: u32,
        max_num_ref_frames: usize,
        meta: Arc<OutputMeta>,
        out: &mut Vec<Output<F>>,
    ) {
        self.close_pending();
        let (mark, long_idx, had5) = self.mark_current(sh, None, None, max_frame_num, max_num_ref_frames);
        if had5 {
            // All earlier pictures are output before the current one (C.4.4).
            self.flush(out);
        }
        let poc = field_poc[0].min(field_poc[1]);
        let frame_num = if had5 { 0 } else { sh.frame_num };
        // Bumping until there is space (C.4.5.1 / C.4.5.2).
        if mark == RefMark::Unused {
            loop {
                let waiting = self.entries.len();
                if waiting < self.capacity {
                    break;
                }
                // non-reference picture with the smallest POC is output directly
                let min_poc = self.entries.iter().filter(|e| e.needed_for_output).map(|e| e.poc).min();
                if min_poc.is_none_or(|m| poc < m) {
                    out.push(Output { frame, poc, meta });
                    return;
                }
                if !self.bump(out) {
                    break;
                }
            }
        } else {
            while self.entries.len() >= self.capacity {
                if !self.bump(out) {
                    break;
                }
            }
        }
        self.entries.push(DpbEntry {
            frame,
            poc,
            frame_num,
            long_term_frame_idx: long_idx,
            mark,
            needed_for_output: true,
            non_existing: false,
            meta,
            field_poc,
            field_mark: [mark; 2],
            fields: 3,
            awaiting_second: false,
        });
        self.output_reordered(out);
    }

    /// Low-delay output: honour max_num_reorder_frames.
    fn output_reordered(&mut self, out: &mut Vec<Output<F>>) {
        while self.entries.iter().filter(|e| e.needed_for_output).count() > self.max_reorder {
            if !self.bump(out) {
                break;
            }
        }
    }

    /// Reference marking of the current picture (8.2.5.1): returns its marking, long-term frame
    /// index and whether MMCO 5 was executed. `field`: parity of a current field; `cur`: entry of
    /// the current frame when the field is a second field.
    fn mark_current(&mut self, sh: &SliceHeader, field: Option<u8>, cur: Option<usize>, max_frame_num: u32, max_num_ref_frames: usize) -> (RefMark, u32, bool) {
        let mut mark = RefMark::Unused;
        let mut long_idx = 0;
        let mut had5 = false;
        if sh.nal_ref_idc == 0 {
            return (mark, long_idx, had5);
        }
        if sh.idr {
            if sh.long_term_reference {
                self.max_long_term_frame_idx = Some(0);
                return (RefMark::Long, 0, false);
            }
            self.max_long_term_frame_idx = None;
            return (RefMark::Short, 0, false);
        }
        let mut cur_long = None;
        // second field of a reference pair whose first field is short-term: no sliding window
        let first_short = cur.and_then(|i| self.entries.get(i)).is_some_and(|e| e.field_mark.contains(&RefMark::Short));
        if sh.adaptive_ref_pic_marking {
            had5 = self.apply_mmcos(&sh.mmcos, sh.frame_num, max_frame_num, field, cur, &mut cur_long);
        } else if !first_short {
            self.sliding_window(sh.frame_num, max_frame_num, max_num_ref_frames);
        }
        if let Some(i) = cur_long {
            mark = RefMark::Long;
            long_idx = i;
        } else {
            mark = RefMark::Short;
            // make room if MMCOs left the DPB over-full (non-conforming streams)
            if cur.is_none() && self.num_refs() >= max_num_ref_frames.max(1) && sh.adaptive_ref_pic_marking {
                self.sliding_window(sh.frame_num, max_frame_num, max_num_ref_frames);
            }
        }
        (mark, long_idx, had5)
    }

    /// Store a decoded field (`parity`, order count `poc`). A second field (`second`) completes the
    /// pending first field's entry; a first field gets a new entry that is not output before its
    /// second field arrives or [`Dpb::close_pending`] is called.
    #[allow(clippy::too_many_arguments)]
    pub fn store_field(
        &mut self,
        sh: &SliceHeader,
        frame: F,
        parity: u8,
        poc: i32,
        second: bool,
        max_frame_num: u32,
        max_num_ref_frames: usize,
        meta: Arc<OutputMeta>,
        out: &mut Vec<Output<F>>,
    ) {
        let cur = if second { self.entries.iter().position(|e| e.awaiting_second) } else { None };
        if !second {
            self.close_pending();
        }
        let (mark, long_idx, had5) = self.mark_current(sh, Some(parity), cur, max_frame_num, max_num_ref_frames);
        if had5 {
            self.close_pending();
            self.flush(out);
        }
        let frame_num = if had5 { 0 } else { sh.frame_num };
        if let Some(i) = cur.filter(|&i| i < self.entries.len() && self.entries[i].awaiting_second) {
            let e = &mut self.entries[i];
            e.fields |= 1 << parity;
            e.field_poc[parity as usize & 1] = poc;
            e.field_mark[parity as usize & 1] = mark;
            if mark == RefMark::Long {
                e.long_term_frame_idx = long_idx;
            }
            e.sync_mark();
            e.poc = e.field_poc[0].min(e.field_poc[1]);
            e.awaiting_second = false;
            self.prune();
            while self.entries.len() > self.capacity {
                if !self.bump(out) {
                    break;
                }
            }
            self.output_reordered(out);
            return;
        }
        // A non-reference first field that precedes everything waiting for output would be output
        // directly as a frame (C.4.5.2): keep it (one buffer over) until its second field arrives.
        let min_poc = self.entries.iter().filter(|e| e.needed_for_output).map(|e| e.poc).min();
        let direct = mark == RefMark::Unused && min_poc.is_none_or(|m| poc < m);
        while !direct && self.entries.len() >= self.capacity {
            if !self.bump(out) {
                break;
            }
        }
        let mut field_mark = [RefMark::Unused; 2];
        field_mark[parity as usize & 1] = mark;
        let mut e = DpbEntry {
            frame,
            poc,
            frame_num,
            long_term_frame_idx: long_idx,
            mark,
            needed_for_output: true,
            non_existing: false,
            meta,
            field_poc: [poc, poc],
            field_mark,
            fields: 1 << parity,
            awaiting_second: true,
        };
        e.sync_mark();
        self.entries.push(e);
    }
}

impl<F: Clone> Default for Dpb<F> {
    fn default() -> Self {
        Self::new()
    }
}
