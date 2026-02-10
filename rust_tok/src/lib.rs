use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;

#[pyclass]
#[derive(Serialize, Deserialize, Debug)]
struct UnigramTok {
    vocab: HashMap<String, f64>,
}

#[pymethods]
impl UnigramTok {
    #[new]
    fn new() -> Self {
        UnigramTok { vocab: HashMap::new() }
    }

    fn normalize(&self, text: &str) -> String {
        text.nfkc().collect()
    }

    fn train(&mut self, file_path: &str, limit: usize) -> PyResult<()> {
        println!("Loading file {}...", file_path);
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        
        let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;
        println!("Starting parallel processing on {} lines...", lines.len());

        let re = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{Han}+| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();

        let counts: HashMap<String, usize> = lines
            .par_iter()
            .fold(
                || HashMap::new(),
                |mut local_map, line| {
                    if line.trim().is_empty() { return local_map; }
                    let norm_line = line.nfkc().collect::<String>();
                    let parts: Vec<&str> = re.find_iter(&norm_line).map(|m| m.as_str()).collect();

                    for part in parts {
                        let chars: Vec<char> = part.chars().collect();
                        let n = chars.len();
                        for i in 0..n {
                            for j in i..std::cmp::min(n, i + 6) {
                                let sub: String = chars[i..=j].into_iter().collect();
                                *local_map.entry(sub).or_insert(0) += 1;
                            }
                        }
                    }
                    local_map
                }
            )
            .reduce(
                || HashMap::new(),
                |mut map1, map2| {
                    for (k, v) in map2 {
                        *map1.entry(k).or_insert(0) += v;
                    }
                    map1
                }
            );

        let total: usize = counts.values().sum();
        let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        
        if sorted.len() > limit { sorted.truncate(limit); }

        self.vocab.clear();
        for (t, c) in sorted {
            self.vocab.insert(t, (c as f64 / total as f64).ln());
        }
        for i in 0..=255u8 {
            let s = (i as char).to_string();
            if !self.vocab.contains_key(&s) {
                self.vocab.insert(s, -100.0);
            }
        }
        
        println!("Training complete. Vocab size: {}", self.vocab.len());
        Ok(())
    }

    fn save(&self, filename: &str) -> PyResult<()> {
        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &self).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    fn load(filename: &str) -> PyResult<Self> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let tok = serde_json::from_reader(reader).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(tok)
    }

    fn encode(&self, text: &str) -> Vec<String> {
        let text = self.normalize(text);
        let re = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{Han}+| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
        let parts: Vec<&str> = re.find_iter(&text).map(|m| m.as_str()).collect();
        let mut res = Vec::new();

        for part in parts {
            res.extend(self.viterbi(part));
        }
        res
    }

    fn viterbi(&self, piece: &str) -> Vec<String> {
        let chars: Vec<char> = piece.chars().collect();
        let n = chars.len();
        let mut dp = vec![-1e9; n + 1];
        let mut parent = vec![None; n + 1];
        dp[0] = 0.0;

        for i in 0..n {
            if dp[i] <= -1e9 { continue; }
            for j in i..std::cmp::min(n, i + 6) {
                let sub: String = chars[i..=j].into_iter().collect();
                if let Some(&score) = self.vocab.get(&sub) {
                    let new_score = dp[i] + score;
                    if new_score > dp[j + 1] {
                        dp[j + 1] = new_score;
                        parent[j + 1] = Some((i, sub));
                    }
                }
            }
        }

        let mut idx = n;
        let mut tokens = Vec::new();
        while idx > 0 {
            match &parent[idx] {
                Some((prev, token)) => {
                    tokens.push(token.clone());
                    idx = *prev;
                },
                None => {
                    tokens.push("<UNK>".to_string());
                    idx -= 1;
                }
            }
        }
        tokens.reverse();
        tokens
    }
}

// ⬇️ THIS IS THE PART THAT CHANGED FOR PYTHON 3.13 ⬇️
#[pymodule]
fn rust_tok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<UnigramTok>()?;
    Ok(())
}