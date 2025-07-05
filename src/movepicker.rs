use arrayvec::ArrayVec;
use crate::board::Board;
use crate::moves::{Move, MoveList};
use crate::thread::ThreadData;
use crate::{movegen, moves, ordering};
use movegen::{gen_moves, MoveFilter};
use moves::MAX_MOVES;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    TTMove,
    GenerateMoves,
    Moves,
    Done
}

pub struct MovePicker {
    moves: MoveList,
    filter: MoveFilter,
    idx: usize,
    stage: Stage,
    tt_move: Move,
    ply: usize,
    bad_noisy: ArrayVec<Move, MAX_MOVES>,
    bad_noisy_idx: usize,
}

impl MovePicker {

    pub fn new(tt_move: Move, filter: MoveFilter, ply: usize) -> Self {
        let stage = if tt_move.exists() { Stage::TTMove } else { Stage::GenerateMoves };
        MovePicker {
            moves: MoveList::new(),
            filter,
            idx: 0,
            stage,
            tt_move,
            ply,
            bad_noisy: ArrayVec::new(),
            bad_noisy_idx: 0,
        }
    }

    pub fn next(&mut self, board: &Board, td: &ThreadData) -> Option<Move> {

        if self.stage == Stage::TTMove {
            self.stage = Stage::GenerateMoves;
            if self.tt_move.exists() {
                return Some(self.tt_move);
            }
        }
        if self.stage == Stage::GenerateMoves {
            self.moves = gen_moves(board, self.filter);
            let scores = ordering::score(td, board, &self.moves, &self.tt_move, self.ply);
            self.moves.sort(&scores);
            self.idx = 0;
            self.stage = Stage::Moves;
        }
        if self.stage == Stage::Moves {
            if self.idx < self.moves.len() {
                let m = self.moves.get(self.idx);
                self.idx += 1;
                if let Some(m) = m {
                    if m != self.tt_move {
                        return Some(m);
                    } else {
                        return self.next(board, td);
                    }
                }
            } else {
                self.stage = Stage::Done;
            }
        }
        None

    }

}