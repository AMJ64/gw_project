use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use serde::{Serialize, Deserialize};
use rayon::prelude::*; // <--- Imports Parallel Iterator

#[derive(Serialize, Deserialize, Debug)]
struct UnigramTok {
    vocab: HashMap<String, f64>,
}

impl UnigramTok {
    fn new() -> Self {
        UnigramTok { vocab: HashMap::new() }
    }

    fn normalize(&self, text: &str) -> String {
        text.nfkc().collect()
    }

    fn pre_tokenize<'a>(&self, text: &'a str, re: &Regex) -> Vec<&'a str> {
        re.find_iter(text).map(|m| m.as_str()).collect()
    }

    // --- OPTIMIZED PARALLEL TRAIN ---
    fn train(&mut self, file_path: &str, limit: usize) -> std::io::Result<()> {
        println!("Loading file {} into memory...", file_path);
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        
        // 1. Load all lines into RAM (Required for Rayon to split them)
        let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;
        println!("File loaded. Starting parallel processing on {} lines...", lines.len());

        // 2. Compile Regex ONCE (The Speed Fix)
        // This is shared across all CPU cores.
        let re = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{Han}+| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();

        // 3. Map-Reduce with Rayon
        let counts: HashMap<String, usize> = lines
            .par_iter() // Parallel Iterator
            .fold(
                || HashMap::new(), // Init thread-local map
                |mut local_map, line| {
                    if line.trim().is_empty() { return local_map; }

                    let norm_line = line.nfkc().collect::<String>();
                    // Pass the shared regex reference
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
                || HashMap::new(), // Init reducer
                |mut map1, map2| {
                    // Merge results from threads
                    for (k, v) in map2 {
                        *map1.entry(k).or_insert(0) += v;
                    }
                    map1
                }
            );

        println!("Counting complete. Pruning vocabulary...");

        // 4. Pruning (Single Threaded is fine here)
        let total: usize = counts.values().sum();
        let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending
        
        if sorted.len() > limit { sorted.truncate(limit); }

        self.vocab.clear();
        for (t, c) in sorted {
            self.vocab.insert(t, (c as f64 / total as f64).ln());
        }

        // 5. Byte Fallback
        for i in 0..=255u8 {
            let s = (i as char).to_string();
            if !self.vocab.contains_key(&s) {
                self.vocab.insert(s, -100.0); // Low probability for raw bytes
            }
        }
        
        Ok(())
    }

    // --- SAVE / LOAD ---
    fn save(&self, filename: &str) -> std::io::Result<()> {
        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }

    fn _load(filename: &str) -> std::io::Result<Self> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let tok = serde_json::from_reader(reader)?;
        Ok(tok)
    }

    // --- INFERENCE ---
    fn encode(&self, text: &str) -> Vec<String> {
        let text = self.normalize(text);
        let re = Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{Han}+| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+").unwrap();
        let parts = self.pre_tokenize(&text, &re);
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

fn main() {
    let data_path = "data.txt";
    let save_path = "tokenizer_multi.json";
    
    // Check if data exists
    if std::fs::metadata(data_path).is_err() {
        eprintln!("Error: 'data.txt' not found.");
        return;
    }

    println!("--- Step 1: Training (Parallel) ---");
    let start = std::time::Instant::now(); 
    
    let mut tok = UnigramTok::new();
    match tok.train(data_path, 8000) {
        Ok(_) => println!("Training complete!"),
        Err(e) => {
            eprintln!("Error: {}", e);
            return;
        }
    }
    
    let duration = start.elapsed();
    println!("⏱️ Training took: {:.2} seconds", duration.as_secs_f64());

    tok.save(save_path).unwrap();

    // Verification
    let sentences = vec![
        "def parallel_code():", 
        "return 'रफ़्तार'",
    ];
    println!("\n--- Test Results ---");
    for s in sentences {
        println!("Input:  {}", s);
        println!("Tokens: {:?}", tok.encode(s)); 
    }
}