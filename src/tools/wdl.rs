use crate::search::score;

pub fn params(mat: i16) -> (f64, f64) {
    let a = [-477.69022982, 1218.76802082, -1040.94178352, 955.44284406];
    let b = [46.64976052, -29.08902812, 11.46635756, 126.45064726];

    let m = mat.clamp(17, 78) as f64 / 58.0;

    (
        ((a[0] * m + a[1]) * m + a[2]) * m + a[3],
        ((b[0] * m + b[1]) * m + b[2]) * m + b[3],
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

pub fn normalize(score: i32, mat: i16) -> i32 {
    if score == 0 || score::is_mate(score){
        return score;
    }

    let a = params(mat).0;
    let normalized = (score as f64) / a;
    score::clamp(f64::round(normalized * 100.0) as i32)
}