use std::fs::File;
use std::io::{BufRead, BufReader};
use hobbes_nnue_arch::arch;
use crate::board::Board;
use crate::evaluation::NNUE;

#[derive(Default)]
struct Stats {
    total: i128,
    count: i128,
    abs_total: i128,
    min: i32,
    max: i32,
}

fn chunk_stats(fens: &[String]) -> Stats {
    let mut stats = Stats::default();

    let mut nnue = NNUE::default();

    for fen in fens {
        let game = Board::from_fen(fen).unwrap();

        nnue.activate(&game);
        let eval = nnue.evaluate(&game);

        stats.count += 1;
        stats.total += i128::from(eval);
        stats.abs_total += i128::from(eval.abs());

        if eval < stats.min {
            stats.min = eval;
        }
        if eval > stats.max {
            stats.max = eval;
        }
    }

    stats
}

fn aggregate_stats(chunk_stats: &[Stats]) -> Stats {
    let mut stats = Stats::default();

    for s in chunk_stats {
        stats.count += s.count;
        stats.total += s.total;
        stats.abs_total += s.abs_total;

        if s.min < stats.min {
            stats.min = s.min;
        }

        if s.max > stats.max {
            stats.max = s.max;
        }
    }

    stats
}

pub fn compute_net_scale(dataset: &str) {
    let file = File::open(dataset).expect("Unable to find dataset");

    let fens = BufReader::new(file)
        .lines()
        .collect::<Result<Vec<_>, _>>().unwrap();

    let stats = fens
        .chunks(100_000)
        .map(chunk_stats)
        .collect::<Vec<_>>();

    let stats = aggregate_stats(&stats);

    println!("Stats:");
    println!("FENs: {:>7}", stats.count);

    let mean = stats.total as f64 / stats.count as f64;
    let abs_mean = stats.abs_total as f64 / stats.count as f64;
    let min = f64::from(stats.min);
    let max = f64::from(stats.max);

    println!("Average: {mean:.2}");
    println!("Average (abs): {abs_mean:.2}");
    println!("Min: {min}");
    println!("Max: {max}");

    // Average eval of hobbes-48, which search is tuned against
    let original_avg = 747.116;

    let scale = original_avg / abs_mean * f64::from(arch::SCALE as i32);

    println!("\nScale: {scale:.6}");
}