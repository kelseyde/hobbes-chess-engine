use crate::board::Board;
use crate::moves::{Move, MoveList};
use crate::thread::ThreadData;
use crate::{movegen, ordering};
use movegen::{gen_moves, MoveFilter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    TTMove,
    GenerateNoisies,
    Noisies,
    GenerateQuiets,
    Quiets,
    Done
}

pub struct MovePicker {
    moves: MoveList,
    filter: MoveFilter,
    idx: usize,
    stage: Stage,
    tt_move: Move,
    ply: usize,
}

impl MovePicker {

    pub fn new(tt_move: Move, filter: MoveFilter, ply: usize) -> Self {
        let stage = if tt_move.exists() { Stage::TTMove } else { Stage::GenerateNoisies };
        MovePicker {
            moves: MoveList::new(),
            filter,
            idx: 0,
            stage,
            tt_move,
            ply
        }
    }

    pub fn next(&mut self, board: &Board, td: &ThreadData) -> Option<Move> {

        if self.stage == Stage::TTMove {
            self.stage = Stage::GenerateNoisies;
            if self.tt_move.exists() {
                return Some(self.tt_move);
            }
        }
        if self.stage == Stage::GenerateNoisies {
            self.moves = gen_moves(board, self.filter);
            let scores = ordering::score(td, board, &self.moves, &self.tt_move, self.ply);
            self.moves.sort(&scores);
            self.idx = 0;
            self.stage = Stage::Noisies;
        }
        if self.stage == Stage::Noisies {
            if self.idx < self.moves.len() {
                let m = self.moves.get(self.idx);
                self.idx += 1;
                if let Some(m) = m {
                    return if m != self.tt_move {
                        Some(m)
                    } else {
                        self.next(board, td)
                    }
                }
            } else {
                self.stage = Stage::Done;
            }
        }
        if self.stage == Stage::GenerateQuiets {
            self.moves = gen_moves(board, MoveFilter::Quiets);
            let scores = ordering::score(td, board, &self.moves, &self.tt_move, self.ply);
            self.moves.sort(&scores);
            self.idx = 0;
            self.stage = Stage::Quiets;
        }
        if self.stage == Stage::Quiets {
            if self.idx < self.moves.len() {
                let m = self.moves.get(self.idx);
                self.idx += 1;
                if let Some(m) = m {
                    return if m != self.tt_move {
                        Some(m)
                    } else {
                        self.next(board, td)
                    }
                }
            } else {
                self.stage = Stage::Done;
            }
        }
        None

    }

}