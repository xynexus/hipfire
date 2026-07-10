use std::path::{Path, PathBuf};

use hipfire_arch_embeddinggemma as embeddinggemma;
use hipfire_rdna::Gpu;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use serde_json::json;

#[derive(Debug)]
struct SelectionMetrics {
    queries: usize,
    pool: usize,
    top1_agreement: f64,
    top5_overlap: f64,
    top10_overlap: f64,
    reference_regret_mean: f64,
    reference_regret_p95: f64,
    reference_regret_max: f64,
    changed_reference_regret_mean: Option<f64>,
    reference_top1_margin_mean: f64,
    flipped_reference_margin_mean: Option<f64>,
    flipped_reference_margin_p95: Option<f64>,
}

struct Pair {
    sentence1: String,
    sentence2: String,
    score: f64,
}

struct PairEmbeddings {
    sentence1: Vec<f32>,
    sentence2: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reference = None;
    let mut candidates = Vec::new();
    let mut dataset = None;
    let mut max_pairs = 128usize;
    let mut selection_queries = 256usize;
    let mut selection_pool = usize::MAX;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--reference" => reference = args.next().map(PathBuf::from),
            "--candidate" => candidates.push(PathBuf::from(
                args.next().ok_or("missing --candidate path")?,
            )),
            "--dataset" => dataset = args.next().map(PathBuf::from),
            "--max-pairs" => max_pairs = args.next().ok_or("missing --max-pairs value")?.parse()?,
            "--selection-queries" => {
                selection_queries = args
                    .next()
                    .ok_or("missing --selection-queries value")?
                    .parse()?
            }
            "--selection-pool" => {
                selection_pool = args
                    .next()
                    .ok_or("missing --selection-pool value")?
                    .parse()?
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: quality_compare --reference BF16.hfq --candidate MODEL.hfq [...] \
                     --dataset sts-dev.tsv [--max-pairs 128] [--selection-queries 256] \
                     [--selection-pool MAX_PAIRS]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    let reference = reference.ok_or("--reference is required")?;
    let dataset = dataset.ok_or("--dataset is required")?;
    if candidates.is_empty() {
        return Err("at least one --candidate is required".into());
    }
    if selection_queries == 0 || selection_pool == 0 {
        return Err("selection query and pool counts must be non-zero".into());
    }

    let pairs = load_sts_pairs(&dataset, max_pairs)?;
    let gold: Vec<f64> = pairs.iter().map(|pair| pair.score).collect();
    let mut gpu = Gpu::init()?;
    eprintln!("encoding reference: {}", reference.display());
    let reference_embeddings = encode_pairs(&reference, &pairs, &mut gpu)?;
    let reference_scores = pair_scores(&reference_embeddings);
    let reference_pearson = pearson(&reference_scores, &gold);
    let reference_spearman = spearman(&reference_scores, &gold);

    for candidate in candidates {
        eprintln!("encoding candidate: {}", candidate.display());
        let candidate_embeddings = encode_pairs(&candidate, &pairs, &mut gpu)?;
        let candidate_scores = pair_scores(&candidate_embeddings);
        let embedding_cosines: Vec<f64> = candidate_embeddings
            .iter()
            .zip(&reference_embeddings)
            .flat_map(|(candidate_pair, reference_pair)| {
                [
                    cosine(&candidate_pair.sentence1, &reference_pair.sentence1),
                    cosine(&candidate_pair.sentence2, &reference_pair.sentence2),
                ]
            })
            .collect();
        let mut sorted_cosines = embedding_cosines.clone();
        sorted_cosines.sort_by(f64::total_cmp);
        let pair_mae = candidate_scores
            .iter()
            .zip(&reference_scores)
            .map(|(candidate_score, reference_score)| (candidate_score - reference_score).abs())
            .sum::<f64>()
            / candidate_scores.len() as f64;
        let candidate_spearman = spearman(&candidate_scores, &gold);
        let selection = selection_stability(
            &candidate_embeddings,
            &reference_embeddings,
            selection_queries,
            selection_pool,
        );
        let (delta_low, delta_high) =
            bootstrap_spearman_delta_ci(&candidate_scores, &reference_scores, &gold, 500);
        let label = candidate
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("candidate");
        println!(
            "{}",
            json!({
                "candidate": candidate,
                "label": label,
                "pairs": pairs.len(),
                "sentences": pairs.len() * 2,
                "reference_pearson": reference_pearson,
                "reference_spearman": reference_spearman,
                "candidate_pearson": pearson(&candidate_scores, &gold),
                "candidate_spearman": candidate_spearman,
                "spearman_delta_vs_reference": candidate_spearman - reference_spearman,
                "spearman_delta_ci95_low": delta_low,
                "spearman_delta_ci95_high": delta_high,
                "pair_cosine_mae_vs_reference": pair_mae,
                "embedding_cosine_mean_vs_reference": mean(&embedding_cosines),
                "embedding_cosine_min_vs_reference": sorted_cosines[0],
                "embedding_cosine_p05_vs_reference": percentile(&sorted_cosines, 0.05),
                "selection_queries": selection.queries,
                "selection_pool": selection.pool,
                "selection_top1_agreement": selection.top1_agreement,
                "selection_top5_overlap": selection.top5_overlap,
                "selection_top10_overlap": selection.top10_overlap,
                "selection_reference_regret_mean": selection.reference_regret_mean,
                "selection_reference_regret_p95": selection.reference_regret_p95,
                "selection_reference_regret_max": selection.reference_regret_max,
                "selection_changed_reference_regret_mean": selection.changed_reference_regret_mean,
                "selection_reference_top1_margin_mean": selection.reference_top1_margin_mean,
                "selection_flipped_reference_margin_mean": selection.flipped_reference_margin_mean,
                "selection_flipped_reference_margin_p95": selection.flipped_reference_margin_p95,
            })
        );
    }
    Ok(())
}

fn load_sts_pairs(path: &Path, max_pairs: usize) -> Result<Vec<Pair>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines
        .next()
        .ok_or("empty STS dataset")?
        .split('\t')
        .collect();
    let sentence1_col = header
        .iter()
        .position(|name| *name == "sentence1")
        .ok_or("missing sentence1 column")?;
    let sentence2_col = header
        .iter()
        .position(|name| *name == "sentence2")
        .ok_or("missing sentence2 column")?;
    let score_col = header
        .iter()
        .position(|name| *name == "score")
        .ok_or("missing score column")?;
    let all: Vec<Pair> = lines
        .filter_map(|line| {
            let columns: Vec<&str> = line.split('\t').collect();
            Some(Pair {
                sentence1: columns.get(sentence1_col)?.to_string(),
                sentence2: columns.get(sentence2_col)?.to_string(),
                score: columns.get(score_col)?.parse().ok()?,
            })
        })
        .collect();
    if all.is_empty() || max_pairs == 0 {
        return Err("STS dataset has no usable pairs".into());
    }
    if all.len() <= max_pairs {
        return Ok(all);
    }
    Ok((0..max_pairs)
        .map(|index| {
            let source_index = index * (all.len() - 1) / (max_pairs - 1).max(1);
            Pair {
                sentence1: all[source_index].sentence1.clone(),
                sentence2: all[source_index].sentence2.clone(),
                score: all[source_index].score,
            }
        })
        .collect())
}

fn encode_pairs(
    path: &Path,
    pairs: &[Pair],
    gpu: &mut Gpu,
) -> Result<Vec<PairEmbeddings>, Box<dyn std::error::Error>> {
    let mut hfq = HfqFile::open(path)?;
    let config = embeddinggemma::config_from_metadata_json(&hfq.metadata_json)
        .ok_or("failed to parse EmbeddingGemma config")?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)?;
    let weights = embeddinggemma::EmbeddingGemmaWeights::load(&mut hfq, &config, gpu)?;
    let mut embeddings = Vec::with_capacity(pairs.len());
    for pair in pairs {
        let sentence1_tokens =
            tokenizer.encode(&format!("{}{}", config.document_prompt, pair.sentence1));
        let sentence2_tokens =
            tokenizer.encode(&format!("{}{}", config.document_prompt, pair.sentence2));
        if sentence1_tokens.len() > config.sliding_window
            || sentence2_tokens.len() > config.sliding_window
        {
            return Err("STS sentence exceeds exact sliding window".into());
        }
        embeddings.push(PairEmbeddings {
            sentence1: embeddinggemma::embed_forward(gpu, &weights, &config, &sentence1_tokens)?,
            sentence2: embeddinggemma::embed_forward(gpu, &weights, &config, &sentence2_tokens)?,
        });
    }
    weights.free_gpu(gpu);
    Ok(embeddings)
}

fn pair_scores(pairs: &[PairEmbeddings]) -> Vec<f64> {
    pairs
        .iter()
        .map(|pair| cosine(&pair.sentence1, &pair.sentence2))
        .collect()
}

fn selection_stability(
    candidate: &[PairEmbeddings],
    reference: &[PairEmbeddings],
    query_count: usize,
    pool_count: usize,
) -> SelectionMetrics {
    assert_eq!(candidate.len(), reference.len());
    let queries = query_count.min(reference.len());
    let pool = pool_count.min(reference.len());
    assert!(queries > 0 && pool > 0);

    let mut top1_matches = 0usize;
    let mut top5_overlap = 0.0;
    let mut top10_overlap = 0.0;
    let mut regrets = Vec::with_capacity(queries);
    let mut changed_regrets = Vec::new();
    let mut reference_margins = Vec::with_capacity(queries);
    let mut flipped_reference_margins = Vec::new();

    for query_index in 0..queries {
        let mut reference_order: Vec<(usize, f64)> = (0..pool)
            .map(|document_index| {
                (
                    document_index,
                    cosine(
                        &reference[query_index].sentence1,
                        &reference[document_index].sentence2,
                    ),
                )
            })
            .collect();
        let mut candidate_order: Vec<(usize, f64)> = (0..pool)
            .map(|document_index| {
                (
                    document_index,
                    cosine(
                        &candidate[query_index].sentence1,
                        &candidate[document_index].sentence2,
                    ),
                )
            })
            .collect();
        reference_order.sort_by(|left, right| right.1.total_cmp(&left.1));
        candidate_order.sort_by(|left, right| right.1.total_cmp(&left.1));

        let reference_top1 = reference_order[0].0;
        let candidate_top1 = candidate_order[0].0;
        let margin = if pool > 1 {
            reference_order[0].1 - reference_order[1].1
        } else {
            0.0
        };
        reference_margins.push(margin);
        let selection_changed = reference_top1 != candidate_top1;
        if !selection_changed {
            top1_matches += 1;
        } else {
            flipped_reference_margins.push(margin);
        }

        let candidate_choice_reference_score = reference_order
            .iter()
            .find(|(document_index, _)| *document_index == candidate_top1)
            .expect("candidate selection must be in reference pool")
            .1;
        let regret = (reference_order[0].1 - candidate_choice_reference_score).max(0.0);
        regrets.push(regret);
        if selection_changed {
            changed_regrets.push(regret);
        }
        top5_overlap += top_k_overlap(&reference_order, &candidate_order, 5);
        top10_overlap += top_k_overlap(&reference_order, &candidate_order, 10);
    }

    regrets.sort_by(f64::total_cmp);
    flipped_reference_margins.sort_by(f64::total_cmp);
    SelectionMetrics {
        queries,
        pool,
        top1_agreement: top1_matches as f64 / queries as f64,
        top5_overlap: top5_overlap / queries as f64,
        top10_overlap: top10_overlap / queries as f64,
        reference_regret_mean: mean(&regrets),
        reference_regret_p95: percentile(&regrets, 0.95),
        reference_regret_max: regrets[regrets.len() - 1],
        changed_reference_regret_mean: (!changed_regrets.is_empty())
            .then(|| mean(&changed_regrets)),
        reference_top1_margin_mean: mean(&reference_margins),
        flipped_reference_margin_mean: (!flipped_reference_margins.is_empty())
            .then(|| mean(&flipped_reference_margins)),
        flipped_reference_margin_p95: (!flipped_reference_margins.is_empty())
            .then(|| percentile(&flipped_reference_margins, 0.95)),
    }
}

fn top_k_overlap(reference: &[(usize, f64)], candidate: &[(usize, f64)], k: usize) -> f64 {
    let k = k.min(reference.len()).min(candidate.len());
    let overlap = reference[..k]
        .iter()
        .filter(|(reference_index, _)| {
            candidate[..k]
                .iter()
                .any(|(candidate_index, _)| candidate_index == reference_index)
        })
        .count();
    overlap as f64 / k as f64
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * percentile).round() as usize]
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    let left_mean = mean(left);
    let right_mean = mean(right);
    let numerator: f64 = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - left_mean) * (right - right_mean))
        .sum();
    let left_norm = left
        .iter()
        .map(|value| (value - left_mean).powi(2))
        .sum::<f64>()
        .sqrt();
    let right_norm = right
        .iter()
        .map(|value| (value - right_mean).powi(2))
        .sum::<f64>()
        .sqrt();
    numerator / (left_norm * right_norm)
}

fn spearman(left: &[f64], right: &[f64]) -> f64 {
    pearson(&ranks(left), &ranks(right))
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let average_rank = (start + end - 1) as f64 / 2.0;
        for index in start..end {
            ranks[order[index]] = average_rank;
        }
        start = end;
    }
    ranks
}

fn bootstrap_spearman_delta_ci(
    candidate: &[f64],
    reference: &[f64],
    gold: &[f64],
    iterations: usize,
) -> (f64, f64) {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut deltas = Vec::with_capacity(iterations);
    let mut candidate_sample = vec![0.0; gold.len()];
    let mut reference_sample = vec![0.0; gold.len()];
    let mut gold_sample = vec![0.0; gold.len()];
    for _ in 0..iterations {
        for sample_idx in 0..gold.len() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let source_idx = state as usize % gold.len();
            candidate_sample[sample_idx] = candidate[source_idx];
            reference_sample[sample_idx] = reference[source_idx];
            gold_sample[sample_idx] = gold[source_idx];
        }
        deltas.push(
            spearman(&candidate_sample, &gold_sample) - spearman(&reference_sample, &gold_sample),
        );
    }
    deltas.sort_by(f64::total_cmp);
    (percentile(&deltas, 0.025), percentile(&deltas, 0.975))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlations_match_perfect_and_reversed_rankings() {
        let ascending = [1.0, 2.0, 3.0, 4.0];
        let descending = [4.0, 3.0, 2.0, 1.0];
        assert!((pearson(&ascending, &ascending) - 1.0).abs() < 1e-12);
        assert!((spearman(&ascending, &descending) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn ranks_average_ties() {
        assert_eq!(ranks(&[3.0, 1.0, 1.0, 2.0]), vec![3.0, 0.5, 0.5, 2.0]);
    }

    #[test]
    fn selection_stability_is_exact_for_identical_embeddings() {
        let embeddings = synthetic_embeddings([1.0, 0.8], [1.0, 0.0]);
        let metrics = selection_stability(&embeddings, &embeddings, 1, 2);
        assert_eq!(metrics.top1_agreement, 1.0);
        assert_eq!(metrics.top5_overlap, 1.0);
        assert_eq!(metrics.reference_regret_mean, 0.0);
    }

    #[test]
    fn selection_regret_distinguishes_near_tie_from_clear_loss() {
        let reference_near = synthetic_embeddings([1.0, 0.999], [1.0, 0.0]);
        let candidate_near = synthetic_embeddings([0.998, 0.999], [1.0, 0.0]);
        let near = selection_stability(&candidate_near, &reference_near, 1, 2);
        assert_eq!(near.top1_agreement, 0.0);
        assert!((near.reference_regret_mean - 0.001).abs() < 1e-6);
        assert!((near.changed_reference_regret_mean.unwrap() - 0.001).abs() < 1e-6);

        let reference_clear = synthetic_embeddings([1.0, 0.0], [1.0, 0.0]);
        let candidate_clear = synthetic_embeddings([0.0, 1.0], [1.0, 0.0]);
        let clear = selection_stability(&candidate_clear, &reference_clear, 1, 2);
        assert_eq!(clear.top1_agreement, 0.0);
        assert_eq!(clear.reference_regret_mean, 1.0);
        assert!(clear.reference_regret_mean > near.reference_regret_mean * 100.0);
    }

    fn synthetic_embeddings(document_scores: [f32; 2], query: [f32; 2]) -> Vec<PairEmbeddings> {
        document_scores
            .into_iter()
            .map(|score| PairEmbeddings {
                sentence1: query.to_vec(),
                sentence2: vec![score, 0.0],
            })
            .collect()
    }
}
