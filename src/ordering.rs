use arrayvec::ArrayVec;
use crate::board::Board;
use crate::moves::{MoveList, MAX_MOVES};

pub const MVV_LVA: [[u8; 7]; 7] = [
    [10, 11, 12, 13, 14, 15, 0], // victim P, attacker K, Q, R, B, N, P, ~
    [20, 21, 22, 23, 24, 25, 0], // victim N, attacker K, Q, R, B, N, P, ~
    [30, 31, 32, 33, 34, 35, 0], // victim B, attacker K, Q, R, B, N, P, ~
    [40, 41, 42, 43, 44, 45, 0], // victim R, attacker K, Q, R, B, N, P, ~
    [50, 51, 52, 53, 54, 55, 0], // victim Q, attacker K, Q, R, B, N, P, ~
    [0, 0, 0, 0, 0, 0, 0],       // victim K, attacker K, Q, R, B, N, P, ~
    [0, 0, 0, 0, 0, 0, 0],       // victim ~, attacker K, Q, R, B, N, P, ~
];

pub fn score(board: &Board, moves: &MoveList) -> [i32; MAX_MOVES] {
    let mut scores = [0; MAX_MOVES];
    let mut idx = 0;
    for m in moves.iter() {
        let victim = board.captured(&m);
        if let Some(v) = victim {
            let attacker = board.piece_at(m.from());
            if let Some(a) = attacker {
                scores[idx] = MVV_LVA[v as usize][a as usize] as i32;
            }
        }
        idx += 1;
    }
    scores
}
