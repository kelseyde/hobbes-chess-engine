use movegen::{gen_moves, MoveFilter};
use crate::board::Board;
use crate::{movegen, ordering};
use crate::moves::{Move, MoveList};
use crate::thread::ThreadData;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    TTMove,
    GenerateMoves,
    Moves,
    Done
}

pub struct MovePicker {
    moves: MoveList,
    index: usize,
    stage: Stage,
    tt_move: Move,
    ply: usize,
}

impl MovePicker {

    pub fn new(tt_move: Move, ply: usize) -> Self {
        let stage = if tt_move.exists() { Stage::TTMove } else { Stage::GenerateMoves };
        MovePicker {
            moves: MoveList::new(),
            index: 0,
            stage,
            tt_move,
            ply,
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
            self.moves = gen_moves(board, MoveFilter::All);
            let scores = ordering::score(td, board, &self.moves, &self.tt_move, self.ply);
            self.moves.sort(&scores);
            self.index = 0;
            self.stage = Stage::Moves;
        }
        if self.stage == Stage::Moves {
            if self.index < self.moves.len() {
                if let Some(m) = self.moves.get(self.index) {
                    self.index += 1;
                    if m != self.tt_move {
                        return Some(m);
                    }
                }
            } else {
                self.stage = Stage::Done;
            }
        }
        None

    }

}