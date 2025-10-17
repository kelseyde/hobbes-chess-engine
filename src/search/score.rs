use crate::search::MAX_PLY;

pub struct Score;

impl Score {
    pub const DRAW: i32 = 0;
    pub const MAX: i32 = 32767;
    pub const MIN: i32 = -32767;
    pub const MATE: i32 = 32766;

    pub const fn is_mate(score: i32) -> bool {
        score.abs() >= Score::MATE - MAX_PLY as i32
    }

    pub const fn is_defined(score: i32) -> bool {
        score >= -Score::MATE && score <= Score::MATE
    }

    pub const fn mate_in(ply: usize) -> i32 {
        Score::MATE - ply as i32
    }

    pub const fn mated_in(ply: usize) -> i32 {
        -Score::MATE + ply as i32
    }

}

pub fn format_score(score: i32) -> String {
    if Score::is_mate(score) {
        let moves = ((Score::MATE - score).max(1) / 2).max(1);
        if score < 0 {
            format!("mate {}", -moves)
        } else {
            format!("mate {}", moves)
        }
    } else {
        format!("cp {}", score)
    }
}