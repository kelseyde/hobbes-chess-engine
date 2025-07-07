use crate::board::Board;
use crate::moves::{Move, MoveList, MAX_MOVES};
use crate::see;
use crate::thread::ThreadData;

pub const TT_MOVE_BONUS: i32 = 1000000;
pub const NOISY_BONUS: i32 = 500000;
pub const KILLER_BONUS: i32 = 250000;
pub const QUIET_BONUS: i32 = 0;

pub fn score(
    td: &ThreadData,
    board: &Board,
    moves: &MoveList,
    tt_move: &Move,
    ply: usize,
) -> [i32; MAX_MOVES] {
    let mut scores = [0; MAX_MOVES];
    for (idx, mv) in moves.iter().enumerate() {
        if tt_move.exists() && mv == tt_move {
            scores[idx] = TT_MOVE_BONUS;
        } else {
            if let Some(victim) = board.captured(mv) {
                if let Some(attacker) = board.piece_at(mv.from()) {
                    let victim_value = see::value(victim);
                    let history_score = td.capture_history.get(board.stm, attacker, mv.to(), victim) as i32;
                    scores[idx] = NOISY_BONUS + victim_value + history_score;
                }
            } else {
                let pc = board.piece_at(mv.from()).unwrap();
                let quiet_score = td.quiet_history.get(board.stm, *mv) as i32;
                let mut cont_score = 0;
                for &prev_ply in &[1] {
                    if prev_ply >= ply {
                        if let (Some(prev_mv), Some(prev_pc)) = (td.ss[ply - prev_ply].mv, td.ss[ply - prev_ply].pc) {
                            cont_score += td.cont_history.get(prev_mv, prev_pc, mv, pc) as i32;
                        }
                    }
                }
                let is_killer = td.ss[ply].killer == Some(*mv);
                let base = if is_killer { KILLER_BONUS } else { QUIET_BONUS };
                scores[idx] = base + quiet_score + cont_score;
            }
        }
    }
    scores
}
