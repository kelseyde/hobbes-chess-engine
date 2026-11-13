use crate::search::score;

/// The WDL model is used to display the expected outcome of a position in terms of probability of
/// winning, losing or drawing, as well as to normalise the displayed score so that 100cp is fitted
/// to a 50% win probability. The model is based on Stockfish's formula and was generated from 20k
/// LTC self-play games.
/// See more: https://github.com/official-stockfish/WDL_model

const A: [f64; 4] = [-477.69022982, 1218.76802082, -1040.94178352, 955.44284406];
const B: [f64; 4] = [46.64976052, -29.08902812, 11.46635756, 126.45064726];

const MIN_MATERIAL: i16 = 17;
const MAX_MATERIAL: i16 = 78;
const MATERIAL_REFERENCE: i16 = 58;

pub fn params(mat: i16) -> (f64, f64) {
    let m = mat.clamp(MIN_MATERIAL, MAX_MATERIAL) as f64 / MATERIAL_REFERENCE as f64;

    (
        ((A[0] * m + A[1]) * m + A[2]) * m + A[3],
        ((B[0] * m + B[1]) * m + B[2]) * m + B[3],
    )
}

pub fn model(score: i32, mat: i16) -> (i16, i16) {
    let (a, b) = params(mat);
    let x = score as f64;

    (
        f64::round(1000.0 / (1.0 + f64::exp((a - x) / b))) as i16,
        f64::round(1000.0 / (1.0 + f64::exp((a + x) / b))) as i16,
    )
}

pub fn normalise(score: i32, mat: i16) -> i32 {
    if score == 0 || score::is_mate(score) {
        return score;
    }

    let a = params(mat).0;
    let normalised = (score as f64) / a;
    score::clamp(f64::round(normalised * 100.0) as i32)
}
