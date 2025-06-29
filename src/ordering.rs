use crate::board::Board;
use crate::moves::{Move, MoveList, MAX_MOVES};
use crate::thread::ThreadData;

pub const TT_MOVE_BONUS: i32 = 1000000;
pub const NOISY_BONUS: i32 = 500000;
pub const QUIET_BONUS: i32 = 0;

pub const MVV_LVA: [[u8; 7]; 7] = [
    [10, 11, 12, 13, 14, 15, 0], // victim P, attacker K, Q, R, B, N, P, ~
    [20, 21, 22, 23, 24, 25, 0], // victim N, attacker K, Q, R, B, N, P, ~
    [30, 31, 32, 33, 34, 35, 0], // victim B, attacker K, Q, R, B, N, P, ~
    [40, 41, 42, 43, 44, 45, 0], // victim R, attacker K, Q, R, B, N, P, ~
    [50, 51, 52, 53, 54, 55, 0], // victim Q, attacker K, Q, R, B, N, P, ~
    [0, 0, 0, 0, 0, 0, 0],       // victim K, attacker K, Q, R, B, N, P, ~
    [0, 0, 0, 0, 0, 0, 0],       // victim ~, attacker K, Q, R, B, N, P, ~
];

pub fn score(td: &ThreadData, board: &Board, moves: &MoveList, tt_move: &Move, ply: usize) -> [i32; MAX_MOVES] {
    let mut scores = [0; MAX_MOVES];
    let mut idx = 0;
    for m in moves.iter() {
        if tt_move.exists() && m == tt_move {
            scores[idx] = TT_MOVE_BONUS;
        } else {
            let victim = board.captured(&m);
            if let Some(v) = victim {
                let attacker = board.piece_at(m.from());
                if let Some(a) = attacker {
                    scores[idx] = NOISY_BONUS + MVV_LVA[v][a] as i32;
                }
            } else {
                let quiet_score = td.quiet_history.get(board.stm, *m) as i32;
                let cont_score = if let Some(prev_mv) = td.ss[ply - 1].mv {
                    let prev_pc = td.ss[ply - 1].pc.unwrap();
                    let pc = board.piece_at(m.from()).unwrap();
                    td.cont_history.get(prev_mv, prev_pc, *m, pc) as i32
                } else {
                    0
                };
                scores[idx] = QUIET_BONUS + quiet_score + cont_score;
            }
        }
        idx += 1;
    }
    scores
}
