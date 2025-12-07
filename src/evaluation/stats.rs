use crate::board::{movegen, Board};
use crate::search::thread::ThreadData;
use std::fs;
use std::path::Path;
use crate::evaluation::network::SCALE;

// Implementation yoinked from Viridithas
pub fn eval_stats(td: &mut ThreadData, input: &Path) {

    let mut total = 0i128;
    let mut count = 0i128;
    let mut abs_total = 0i128;
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    let mut sq_total = 0i128;

    let binding = fs::read_to_string(input).unwrap();
    let lines = binding.lines();
    let file_len = lines.clone().count();
    for (i, line) in lines.into_iter().enumerate() {

        let parts: Vec<&str> = line.split("[").collect();
        let fen = parts[0].trim();
        let board = Board::from_fen(fen).unwrap();
        td.nnue.activate(&board);
        if movegen::is_check(&board, board.stm) {
            continue;
        }
        let eval = td.nnue.evaluate(&board);

        count += 1;
        total += i128::from(eval);
        abs_total += i128::from(eval.abs());
        sq_total += i128::from(eval) * i128::from(eval);
        if eval < min {
            min = eval;
        }
        if eval > max {
            max = eval;
        }

        if i % 1024 == 0 {
            print!("\rPROCESSED {:>10}/{}.", i + 1, file_len);
        }

    }

    println!("\rPROCESSED {file_len:>10}/{file_len}.");

    println!(" EVALUATION STATISTICS:");

    println!("    COUNT: {count:>7}");
    #[expect(clippy::cast_precision_loss)]
    if count > 0 {
        let mean = total as f64 / count as f64;
        let abs_mean = abs_total as f64 / count as f64;
        let mean_squared = mean * mean;
        let variance = (sq_total as f64 / count as f64) - mean_squared;
        let stddev = variance.sqrt();
        let min = f64::from(min);
        let max = f64::from(max);
        println!("     MEAN: {mean:>10.2}");
        println!(" ABS MEAN: {abs_mean:>10.2}");
        println!("   STDDEV: {stddev:>10.2}");
        println!("      MIN: {min:>10.2}");
        println!("      MAX: {max:>10.2}");

        // Compute the target scaling factor to achieve the same absolute mean as the master network
        let master_abs_mean = 1233.83;
        let scale = master_abs_mean / abs_mean * f64::from(SCALE);
        println!("  TARGET SCALING FACTOR: {scale:.6}");
    }

}