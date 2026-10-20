//! Interlaced H.264: field pictures (PAFF) and MBAFF frames.
//!
//! PAFF streams come from our own encoder run as two field encoders (one per parity, see
//! `filmcraft_h264enc::EncoderConfig::field_parity`) whose access units are interleaved: the decoded
//! frames must equal the two encoders' reconstructions interleaved line by line, bit for bit.

use filmcraft_h264::{Decoder, Picture};
use filmcraft_h264enc::{Encoder, EncoderConfig, FrameType, PacketFormat, Profile, RateControl, ReconFrame, YuvFrame};

/// A deterministic interlaced test sequence: texture, a gradient and objects moving at different
/// speeds, sampled at field times (the bottom field is half a frame later), 4:2:0.
fn source(w: usize, h: usize, n: usize) -> Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let (cw, ch) = (w / 2, h / 2);
    (0..n)
        .map(|t| {
            let mut y = vec![0u8; w * h];
            let mut u = vec![0u8; cw * ch];
            let mut v = vec![0u8; cw * ch];
            for yy in 0..h {
                // field time: bottom lines are sampled half a frame later
                let ft = 2 * t + (yy & 1);
                for xx in 0..w {
                    let tex = ((xx * 7 + yy * 3) ^ (xx * yy / 5)) & 31;
                    let mut val = 40 + (xx * 120 / w) + tex;
                    let bx = (ft * 3 + 10) % w;
                    if xx >= bx && xx < bx + 24 && yy >= h / 4 && yy < h / 4 + 30 {
                        val = 230 - tex;
                    }
                    let by = (ft * 2 + 5) % h;
                    if yy >= by && yy < by + 12 && xx >= w / 2 && xx < w / 2 + 40 {
                        val = 20 + tex;
                    }
                    y[yy * w + xx] = val.min(255) as u8;
                }
            }
            for yy in 0..ch {
                for xx in 0..cw {
                    u[yy * cw + xx] = (90 + (xx * 60 / cw) + ((t * 2 + yy) & 15)) as u8;
                    v[yy * cw + xx] = (160 - (yy * 50 / ch) + ((xx + t) & 7)) as u8;
                }
            }
            (y, u, v)
        })
        .collect()
}

/// Lines of `parity` of a plane `w` wide.
fn field_of(p: &[u8], w: usize, parity: usize) -> Vec<u8> {
    p.chunks_exact(w).skip(parity).step_by(2).flatten().copied().collect()
}

struct Paff {
    stream: Vec<(Vec<u8>, i64)>,
    /// Expected decoded frames by pts.
    frames: Vec<(i64, Vec<u8>, Vec<u8>, Vec<u8>)>,
    types: Vec<FrameType>,
}

/// Encode `frames` as a PAFF stream: field `first` (0 = top first) then the other one.
fn encode_paff(w: usize, h: usize, frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)], profile: Profile, bframes: u8, first: u8) -> Paff {
    let mut encs: Vec<Encoder> = (0..2u8)
        .map(|k| {
            let mut cfg = EncoderConfig::new(w as u32, h as u32 / 2, 25, 1);
            cfg.profile = profile;
            cfg.rate = RateControl::Qp(22);
            cfg.bframes = bframes;
            cfg.threads = 1;
            cfg.slices = 1;
            cfg.scenecut = false;
            cfg.aq_strength = 0.0;
            cfg.keyint = 1000;
            cfg.aud = false;
            cfg.format = PacketFormat::AnnexB;
            cfg.field_parity = Some(first ^ k);
            cfg.field_second = k == 1;
            let mut e = Encoder::new(cfg).unwrap();
            e.set_recon_capture(true);
            e
        })
        .collect();
    let mut pkts = [Vec::new(), Vec::new()];
    for (i, (y, u, v)) in frames.iter().enumerate() {
        for k in 0..2 {
            let p = (first as usize) ^ k;
            let (fy, fu, fv) = (field_of(y, w, p), field_of(u, w / 2, p), field_of(v, w / 2, p));
            let f = YuvFrame { y: &fy, u: &fu, v: &fv, y_stride: w, uv_stride: w / 2 };
            pkts[k].extend(encs[k].encode(&f, i as i64).unwrap());
        }
    }
    let mut recon: [Vec<ReconFrame>; 2] = [Vec::new(), Vec::new()];
    for k in 0..2 {
        pkts[k].extend(encs[k].flush());
        recon[k] = encs[k].take_recon();
    }
    assert_eq!(pkts[0].len(), pkts[1].len());
    let types: Vec<FrameType> = pkts[0].iter().map(|p| p.frame_type).collect();
    let types1: Vec<FrameType> = pkts[1].iter().map(|p| p.frame_type).collect();
    assert_eq!(types, types1, "both field encoders must make the same picture type decisions");
    let mut stream = Vec::new();
    for (a, b) in pkts[0].iter().zip(&pkts[1]) {
        assert_eq!(a.pts, b.pts);
        stream.push((a.data.clone(), a.pts));
        stream.push((b.data.clone(), a.pts));
    }
    // expected frames: interleave the reconstructions of the two fields
    let mut out = Vec::new();
    for r0 in &recon[0] {
        let r1 = recon[1].iter().find(|r| r.pts == r0.pts).expect("second field reconstruction");
        let mut planes = [Vec::new(), Vec::new(), Vec::new()];
        for (c, (a, b)) in [(&r0.y, &r1.y), (&r0.u, &r1.u), (&r0.v, &r1.v)].into_iter().enumerate() {
            let cw = if c == 0 { w } else { w / 2 };
            let (top, bottom) = if first == 0 { (a, b) } else { (b, a) };
            for (lt, lb) in top.chunks_exact(cw).zip(bottom.chunks_exact(cw)) {
                planes[c].extend_from_slice(lt);
                planes[c].extend_from_slice(lb);
            }
        }
        let [y, u, v] = planes;
        out.push((r0.pts, y, u, v));
    }
    out.sort_by_key(|f| f.0);
    Paff { stream, frames: out, types }
}

fn decode_all(stream: &[(Vec<u8>, i64)], threads: usize) -> Vec<Picture> {
    let mut dec = Decoder::with_threads(threads);
    let mut pics = Vec::new();
    for (au, pts) in stream {
        pics.extend(dec.decode(au, *pts).unwrap_or_else(|e| panic!("decode error at pts {pts}: {e}")));
    }
    pics.extend(dec.flush());
    assert!(dec.take_error().is_none());
    pics
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.len().max(1) as f64;
    if mse == 0.0 { 99.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
}

fn check_paff(profile: Profile, bframes: u8, first: u8, threads: usize) {
    let (w, h, n) = (176, 128, 14);
    let src = source(w, h, n);
    let paff = encode_paff(w, h, &src, profile, bframes, first);
    if bframes > 0 {
        assert!(paff.types.contains(&FrameType::B), "expected B fields: {:?}", paff.types);
    }
    let pics = decode_all(&paff.stream, threads);
    assert_eq!(pics.len(), n, "one frame per field pair");
    for (pic, (pts, y, u, v)) in pics.iter().zip(&paff.frames) {
        assert_eq!(pic.pts, *pts, "output order");
        assert!(pic.interlaced);
        assert_eq!((pic.width as usize, pic.height as usize), (w, h));
        let ctx = format!("{profile:?} bframes {bframes} first field {first} threads {threads}: frame pts {pts}");
        assert!(pic.y == *y, "{ctx}: luma differs from the encoder's reconstruction (PSNR {:.2} dB)", psnr(&pic.y, y));
        assert!(pic.u == *u && pic.v == *v, "{ctx}: chroma differs from the encoder's reconstruction");
        // and the reconstruction is a faithful coding of the source (no field mix-up)
        let s = &src[*pts as usize];
        assert!(psnr(&pic.y, &s.0) > 34.0, "{ctx}: PSNR vs source {:.2}", psnr(&pic.y, &s.0));
    }
}

#[test]
fn paff_cavlc_ip_top_first() {
    check_paff(Profile::Baseline, 0, 0, 1);
}

#[test]
fn paff_cavlc_ip_bottom_first() {
    check_paff(Profile::Baseline, 0, 1, 1);
}

#[test]
fn paff_cabac_b_fields() {
    check_paff(Profile::Main, 2, 0, 1);
}

#[test]
fn paff_high_8x8_b_fields_threaded() {
    check_paff(Profile::High, 2, 0, 4);
    check_paff(Profile::High, 1, 1, 3);
}

/// A field without its second field (the stream ends after it) is output as a frame whose
/// missing lines repeat the decoded field; nothing hangs and the preceding frames are exact.
#[test]
fn paff_unpaired_last_field() {
    let (w, h, n) = (176, 128, 5);
    let src = source(w, h, n);
    let paff = encode_paff(w, h, &src, Profile::Main, 0, 0);
    let mut stream = paff.stream.clone();
    stream.pop();
    for threads in [1, 4] {
        let pics = decode_all(&stream, threads);
        assert_eq!(pics.len(), n);
        for (pic, (_, y, _, _)) in pics.iter().zip(&paff.frames).take(n - 1) {
            assert!(pic.y == *y);
        }
        let last = pics.last().unwrap();
        let top = field_of(&last.y, w, 0);
        assert_eq!(top, field_of(&paff.frames[n - 1].1, w, 0), "decoded field");
        assert_eq!(field_of(&last.y, w, 1), top, "missing field repeats the decoded one");
    }
}
