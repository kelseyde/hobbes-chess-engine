use crate::search::MAX_PLY;
use crate::search::thread::ThreadData;

pub const MAX: i32 = 32767;
pub const MIN: i32 = -32767;
pub const MATE: i32 = 32766;

/// Returns true if the score is a mate-or-mated score (within MAX_PLY of the nominal mate value).
#[inline]
pub const fn is_mate(score: i32) -> bool {
    score.abs() >= MATE - MAX_PLY as i32
}

/// Returns true if the score indicates the side to move is being mated.
#[inline]
pub const fn is_mated(score: i32) -> bool {
    score <= -MATE + MAX_PLY as i32
}

/// Returns true if the score is a real (non-sentinel) value.
#[inline]
pub const fn is_defined(score: i32) -> bool {
    score >= -MATE && score <= MATE
}

/// The score for delivering mate in `ply` half-moves from the root.
#[inline]
pub const fn mate_in(ply: usize) -> i32 {
    MATE - ply as i32
}

/// The score for being mated in `ply` half-moves from the root.
#[inline]
pub const fn mated_in(ply: usize) -> i32 {
    -MATE + ply as i32
}

pub const fn draw(td: &ThreadData) -> i32 {
    (td.nodes % 5) as i32 - 2
}

/// Clamp a score to the valid [MIN, MAX] range.
#[inline]
pub const fn clamp(score: i32) -> i32 {
    if score < MIN {
        MIN
    } else if score > MAX {
        MAX
    } else {
        score
    }
}

/// Adjust a mate score from search space into TT storage space.
/// Non-mate scores are stored as-is. Mate scores are stored relative to the
/// current position rather than the root, so they remain valid on retrieval.
#[inline]
pub const fn to_tt(score: i32, ply: usize) -> i32 {
    if !is_mate(score) {
        score
    } else if score > 0 {
        score - ply as i32
    } else {
        score + ply as i32
    }
}

/// Adjust a mate score from TT storage space back into search space.
/// This is the inverse of `to_tt`.
#[inline]
pub const fn to_search(score: i32, ply: usize) -> i32 {
    if !is_mate(score) {
        score
    } else if score > 0 {
        score + ply as i32
    } else {
        score - ply as i32
    }
}

/// Format a score for UCI output, either as `cp <centipawns>` or `mate <n>`.
pub fn format_score(score: i32) -> String {
    if is_mate(score) {
        // Distance in full moves to mate; positive = we give mate, negative = we are mated.
        let plies = MATE - score.abs();
        let moves = (plies + 1) / 2;
        if score < 0 {
            format!("mate -{moves}")
        } else {
            format!("mate {moves}")
        }
    } else {
        format!("cp {score}")
    }
}
