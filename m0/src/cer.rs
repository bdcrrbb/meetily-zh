// Normalized CER: char-level, with documented zh normalization.
use anyhow::Result;
use serde::Serialize;

fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        // full-width → ascii (FF01..FF5E → 21..7E)
        let c = if ('\u{FF01}'..='\u{FF5E}').contains(&c) {
            char::from_u32(c as u32 - 0xFEE0).unwrap_or(c)
        } else {
            c
        };
        if c.is_whitespace() {
            continue; // spaces removed (Han-Latin spacing policy)
        }
        if c.is_ascii_punctuation() || is_zh_punct(c) {
            continue; // punctuation removed
        }
        for lc in c.to_lowercase() {
            out.push(lc);
        }
    }
    out
}

fn is_zh_punct(c: char) -> bool {
    matches!(
        c,
        '，' | '。' | '、' | '；' | '：' | '？' | '！' | '“' | '”' | '‘' | '’'
            | '（' | '）' | '《' | '》' | '…' | '—' | '·' | '～' | '【' | '】'
    )
}

fn cer(ref_: &str, hyp: &str) -> (usize, usize) {
    // char-level Levenshtein distance over normalized strings
    let r: Vec<char> = normalize(ref_).chars().collect();
    let h: Vec<char> = normalize(hyp).chars().collect();
    let dist = strsim::levenshtein(&r.iter().collect::<String>(), &h.iter().collect::<String>());
    (dist, r.len())
}

#[derive(Serialize)]
struct ClipCer {
    clip: String,
    distance: usize,
    ref_chars: usize,
    cer: f64,
}

#[derive(Serialize)]
struct Out {
    normalization: &'static str,
    clips: Vec<ClipCer>,
    pooled_distance: usize,
    pooled_ref_chars: usize,
    pooled_cer: f64,
}

pub fn run(refs: &str, hyps: &str, out: &str) -> Result<()> {
    let refs: std::collections::HashMap<String, String> =
        serde_json::from_str(&std::fs::read_to_string(refs)?)?;
    let hyps: std::collections::HashMap<String, String> =
        serde_json::from_str(&std::fs::read_to_string(hyps)?)?;
    let mut clips = Vec::new();
    let mut pd = 0usize;
    let mut pn = 0usize;
    for (clip, r) in &refs {
        let h = hyps.get(clip).ok_or_else(|| anyhow::anyhow!("no hyp for {clip}"))?;
        let (d, n) = cer(r, h);
        pd += d;
        pn += n;
        clips.push(ClipCer {
            clip: clip.clone(),
            distance: d,
            ref_chars: n,
            cer: if n == 0 { 0.0 } else { d as f64 / n as f64 },
        });
    }
    let out_v = Out {
        normalization: "lowercase; whitespace removed; ascii+zh punctuation removed; fullwidth->halfwidth",
        clips,
        pooled_distance: pd,
        pooled_ref_chars: pn,
        pooled_cer: if pn == 0 { 0.0 } else { pd as f64 / pn as f64 },
    };
    for c in &out_v.clips {
        println!("{}: cer={:.4} (d={} n={})", c.clip, c.cer, c.distance, c.ref_chars);
    }
    println!("POOLED CER: {:.4} ({} / {})", out_v.pooled_cer, pd, pn);
    std::fs::write(out, serde_json::to_string_pretty(&out_v)?)?;
    Ok(())
}
