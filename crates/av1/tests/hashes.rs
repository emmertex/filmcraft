//! Bit-exactness pins: decoded-output hashes of libaom conformance vectors (downloaded on first
//! use), single- and multi-threaded. These need no oracle decoder, so they guard optimisations on
//! any host. `AV1_HASH_PRINT=1` prints the current hashes instead of asserting.

mod common;
use common::*;
use filmcraft_av1::Decoder;

const PINS: &[(&str, u64)] = &[
    ("av1-1-b8-02-allintra.ivf", 0x7eb89de0771a5bd8),
    ("av1-1-b8-05-mv.ivf", 0xd42fd668889df0ae),
    ("av1-1-b8-06-mfmv.ivf", 0xfef72b7843e5a5e7),
    ("av1-1-b8-04-cdfupdate.ivf", 0x58b21f1a180e1290),
    ("av1-1-b8-00-quantizer-00.ivf", 0x482aa2eef04ddd96),
    ("av1-1-b8-00-quantizer-31.ivf", 0x268affbde1f48d74),
    ("av1-1-b10-00-quantizer-40.ivf", 0xdf49c6617849c707),
    ("av1-1-b8-01-size-66x66.ivf", 0xe01e7c00d691359a),
    ("av1-1-b8-22-svc-L2T2.ivf", 0x4ce389d69bb57d12),
    ("av1-1-b8-16-intra_only-intrabc-extreme-dv.ivf", 0x51a73af27f10c412),
    ("av1-1-b8-24-monochrome.ivf", 0x4b75bdc84517bfa8),
    ("av1-1-b10-23-film_grain-50.ivf", 0x11b55f7245e2f186),
];

fn fnv(h: &mut u64, v: u64) {
    for b in v.to_le_bytes() {
        *h = (*h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
}

fn decode_hash(path: &std::path::Path, threads: usize) -> u64 {
    let mut dec = Decoder::with_threads(threads);
    let mut pics = Vec::new();
    for f in ivf_frames(path) {
        pics.extend(dec.decode(&f).unwrap());
    }
    pics.extend(dec.flush());
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in &pics {
        fnv(&mut h, ((p.width as u64) << 32) | p.height as u64);
        for plane in &p.planes {
            fnv(&mut h, plane.len() as u64);
            for &v in plane {
                h = (h ^ v as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    h
}

#[test]
fn decoded_hashes_are_pinned() {
    let print = std::env::var("AV1_HASH_PRINT").is_ok();
    // Any local IVF (e.g. a 1080p clip) for before/after comparisons by hand.
    if let Ok(f) = std::env::var("AV1_HASH_FILE") {
        let path = std::path::Path::new(&f);
        println!("{f}: {:#018x} (1 thread) {:#018x} (4 threads)", decode_hash(path, 1), decode_hash(path, 4));
    }
    for &(name, want) in PINS {
        let Some(path) = test_vector(name) else { continue };
        for threads in [1, 4] {
            let got = decode_hash(&path, threads);
            if print {
                println!("    (\"{name}\", {got:#018x}), // threads {threads}");
            } else {
                assert_eq!(got, want, "{name} with {threads} threads");
            }
        }
    }
}
