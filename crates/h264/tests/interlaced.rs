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

// ------------------------------------------------------------------------------------------------
// MBAFF (and interlaced frame pictures) from libx264, checked against x264's own reconstruction
// ------------------------------------------------------------------------------------------------

mod common;

/// One libx264 interlaced encode: the stream and the encoder's reconstruction (`dump-yuv`).
struct X264Case {
    name: &'static str,
    width: u32,
    height: u32,
    frames: u32,
    /// lavfi source filter graph (interlaced content: two fields from different times).
    source: &'static str,
    /// ffmpeg encoder options.
    args: &'static [&'static str],
    /// Extra x264 parameters (`key=value:...`).
    params: &'static str,
}

const MOTION: &str = "testsrc2=size={W}x{H}:rate=50,tinterlace=mode=interleave_top";
const NOISY: &str = "mandelbrot=size={W}x{H}:rate=50,noise=alls=10:allf=t,tinterlace=mode=interleave_top";

/// Generate (once) the stream and the reconstruction of `c` in the fixture directory; None (after
/// reporting a skip) without ffmpeg / libx264.
fn x264_fixture(c: &X264Case) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let ff = filmcraft_testkit::ffmpeg_or_skip(c.name)?;
    let dir = common::fixtures_dir();
    let stream = dir.join(format!("{}.h264", c.name));
    let recon = dir.join(format!("{}.recon.yuv", c.name));
    if std::fs::metadata(&stream).is_ok_and(|m| m.len() > 0) && std::fs::metadata(&recon).is_ok_and(|m| m.len() > 0) {
        return Some((stream, recon));
    }
    let src = c.source.replace("{W}", &c.width.to_string()).replace("{H}", &c.height.to_string());
    // x264's parameter list is ':'-separated, so the dump goes to a relative path
    let tmp_recon = filmcraft_testkit::fixtures::temp_path(&recon);
    let tmp_stream = filmcraft_testkit::fixtures::temp_path(&stream);
    let rel = tmp_recon.file_name()?.to_string_lossy().to_string();
    let mut params = format!("dump-yuv={rel}");
    if !c.params.is_empty() {
        params = format!("{}:{params}", c.params);
    }
    let ok = std::process::Command::new(&ff)
        .current_dir(&dir)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", &src, "-frames:v", &c.frames.to_string(), "-pix_fmt", "yuv420p"])
        .args(["-c:v", "libx264", "-threads", "1"])
        .args(c.args)
        .args(["-x264-params", &params, "-f", "h264"])
        .arg(&tmp_stream)
        .status()
        .is_ok_and(|s| s.success());
    if !ok || !tmp_recon.exists() {
        let _ = std::fs::remove_file(&tmp_recon);
        let _ = std::fs::remove_file(&tmp_stream);
        filmcraft_testkit::skip(c.name, "libx264 encode with dump-yuv failed");
        return None;
    }
    std::fs::rename(&tmp_stream, &stream).ok()?;
    std::fs::rename(&tmp_recon, &recon).ok()?;
    Some((stream, recon))
}

/// Decode `stream` with `threads` workers.
fn decode_annexb(stream: &[u8], threads: usize) -> Vec<Picture> {
    let mut dec = Decoder::with_threads(threads);
    let mut pics = Vec::new();
    for (i, au) in common::split_access_units(stream).into_iter().enumerate() {
        pics.extend(dec.decode(au, i as i64).unwrap_or_else(|e| panic!("decode error at access unit {i}: {e}")));
    }
    pics.extend(dec.flush());
    assert!(dec.take_error().is_none());
    pics
}

/// Every decoded frame equals x264's reconstruction bit for bit (single- and multi-threaded).
fn check_x264(c: &X264Case) -> Option<filmcraft_h264::DecodeStats> {
    let (stream, recon) = x264_fixture(c)?;
    let data = std::fs::read(&stream).unwrap();
    let recon = std::fs::read(&recon).unwrap();
    let (w, h) = (c.width as usize, c.height as usize);
    let fsize = w * h * 3 / 2;
    assert_eq!(recon.len(), fsize * c.frames as usize, "{}: reconstruction size", c.name);
    let mut stats = None;
    for threads in [1, 4] {
        let pics = decode_annexb(&data, threads);
        assert_eq!(pics.len(), c.frames as usize, "{}: frame count", c.name);
        for (i, p) in pics.iter().enumerate() {
            assert!(p.interlaced, "{}: frame {i} flagged interlaced", c.name);
            assert_eq!((p.width as usize, p.height as usize), (w, h));
            let r = &recon[i * fsize..(i + 1) * fsize];
            let (ry, ruv) = r.split_at(w * h);
            let (ru, rv) = ruv.split_at(w * h / 4);
            if p.y != ry || p.u != ru || p.v != rv {
                let first = p.y.iter().zip(ry).position(|(a, b)| a != b);
                panic!(
                    "{} ({threads} threads): frame {i} differs from x264's reconstruction (PSNR {:.2} dB, first luma difference at {:?})",
                    c.name,
                    psnr(&p.y, ry),
                    first.map(|o| (o % w, o / w))
                );
            }
        }
        if threads == 1 {
            let mut d = Decoder::with_threads(1);
            for au in common::split_access_units(&data) {
                d.decode(au, 0).unwrap();
            }
            d.flush();
            stats = Some(d.stats());
        }
    }
    stats
}

#[test]
fn mbaff_cavlc_p_b() {
    check_x264(&X264Case {
        name: "mbaff_cavlc",
        width: 352,
        height: 288,
        frames: 12,
        source: MOTION,
        args: &["-flags", "+ildct", "-coder", "0", "-qp", "26", "-bf", "2"],
        params: "tff=1",
    });
}

#[test]
fn mbaff_cabac_pyramid_multiref_weighted() {
    let stats = check_x264(&X264Case {
        name: "mbaff_cabac",
        width: 352,
        height: 288,
        frames: 16,
        source: NOISY,
        args: &["-flags", "+ildct", "-qp", "24", "-bf", "3", "-refs", "4"],
        params: "tff=1:weightp=2:b-pyramid=normal",
    });
    if let Some(s) = stats {
        // the stream exercises field and frame macroblock pairs, B and weighted P pictures
        assert!(s.slices_b > 0 && s.weighted_slices > 0 && s.mb_inter > 0 && s.mb_b_skip > 0, "{s:?}");
    }
}

#[test]
fn mbaff_temporal_direct_8x8() {
    check_x264(&X264Case {
        name: "mbaff_temporal",
        width: 352,
        height: 288,
        frames: 14,
        source: MOTION,
        args: &["-flags", "+ildct", "-qp", "22", "-bf", "3", "-refs", "3"],
        params: "tff=1:direct=temporal:8x8dct=1",
    });
}

#[test]
fn mbaff_bottom_field_first_slices_deblock_offsets() {
    check_x264(&X264Case {
        name: "mbaff_bff_slices",
        width: 352,
        height: 288,
        frames: 12,
        source: NOISY,
        args: &["-flags", "+ildct", "-qp", "34", "-bf", "2", "-slices", "3"],
        params: "bff=1:deblock=2,-1",
    });
}

/// 1080i: cropped (1088 coded lines), CAVLC, deblocking at low QP.
#[test]
fn mbaff_1080i_cropped() {
    check_x264(&X264Case {
        name: "mbaff_1080i",
        width: 1920,
        height: 1080,
        frames: 4,
        source: NOISY,
        args: &["-flags", "+ildct", "-coder", "0", "-qp", "30", "-bf", "1"],
        params: "tff=1",
    });
}

/// Frame pictures of an interlaced stream without MBAFF (x264's fake-interlaced mode).
#[test]
fn interlaced_frame_pictures_without_mbaff() {
    check_x264(&X264Case {
        name: "fake_interlaced",
        width: 352,
        height: 288,
        frames: 10,
        source: MOTION,
        args: &["-qp", "26", "-bf", "2"],
        params: "fake-interlaced=1",
    });
}

/// Corrupted interlaced streams (bit flips, truncated and dropped access units, PAFF and MBAFF)
/// never panic or hang, single- and multi-threaded.
#[test]
fn corrupt_interlaced_streams_never_panic() {
    let (w, h) = (176, 128);
    let paff = encode_paff(w, h, &source(w, h, 8), Profile::Main, 2, 0);
    let mut streams: Vec<Vec<Vec<u8>>> = vec![paff.stream.iter().map(|(d, _)| d.clone()).collect()];
    for name in ["mbaff_cabac", "mbaff_cavlc"] {
        let f = common::fixtures_dir().join(format!("{name}.h264"));
        if let Ok(data) = std::fs::read(&f) {
            streams.push(common::split_access_units(&data).into_iter().map(|a| a.to_vec()).collect());
        }
    }
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for (si, aus) in streams.iter().enumerate() {
        for iter in 0..40 {
            let mut aus = aus.clone();
            match iter % 4 {
                0 => {
                    // drop an access unit (e.g. one field of a pair)
                    let i = 1 + next() as usize % (aus.len() - 1);
                    aus.remove(i);
                }
                1 => {
                    // truncate one
                    let i = next() as usize % aus.len();
                    let n = aus[i].len();
                    aus[i].truncate(n / 2 + next() as usize % (n / 2 + 1));
                }
                _ => {
                    for _ in 0..1 + iter % 5 {
                        let i = next() as usize % aus.len();
                        let n = aus[i].len();
                        if n > 6 {
                            let b = 5 + next() as usize % (n - 5);
                            aus[i][b] ^= 1 << (next() % 8);
                        }
                    }
                }
            }
            for threads in [1, 3] {
                let r = std::panic::catch_unwind(|| {
                    let mut dec = Decoder::with_threads(threads);
                    let mut n = 0;
                    for (k, au) in aus.iter().enumerate() {
                        match dec.decode(au, k as i64) {
                            Ok(p) => n += p.len(),
                            // a panic inside a decoding job is caught and reported as this error
                            Err(e) => assert!(!e.to_string().contains("internal error"), "{e}"),
                        }
                    }
                    if let Some(e) = dec.take_error() {
                        assert!(!e.to_string().contains("internal error"), "{e}");
                    }
                    n + dec.flush().len()
                });
                assert!(r.is_ok(), "stream {si} mutation {iter} ({threads} threads) panicked");
            }
        }
    }
}
