use crate::board::Board;
use crate::moves::{Move, MoveList, MoveListEntry};
use crate::thread::ThreadData;
use crate::{movegen, see};
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
    pub skip_quiets: bool,
}

impl MovePicker {

    pub fn new(tt_move: Move, ply: usize) -> Self {
        let stage = if tt_move.exists() { Stage::TTMove } else { Stage::GenerateNoisies };
        Self {
            moves: MoveList::new(),
            filter: MoveFilter::Noisies,
            idx: 0,
            stage,
            tt_move,
            ply,
            skip_quiets: false,
        }
    }

    pub fn new_qsearch(tt_move: Move, filter: MoveFilter, ply: usize) -> Self {
        let stage = if tt_move.exists() { Stage::TTMove } else { Stage::GenerateNoisies };
        Self {
            moves: MoveList::new(),
            filter,
            idx: 0,
            stage,
            tt_move,
            ply,
            skip_quiets: true,
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
            self.generate(board, td, self.filter, Stage::Noisies);
        }
        if self.stage == Stage::Noisies {
            if let Some(best_move) = self.pick() {
                return Some(best_move)
            } else {
                self.stage = Stage::GenerateQuiets;
            }
        }
        if self.stage == Stage::GenerateQuiets {
            if self.skip_quiets {
                self.stage = Stage::Done;
                return None;
            }
            self.generate(board, td, MoveFilter::Quiets, Stage::Quiets);
        }
        if self.stage == Stage::Quiets {
            if self.skip_quiets {
                self.stage = Stage::Done;
                return None;
            }
            return if let Some(best_move) = self.pick() {
                Some(best_move)
            } else {
                None
            }
        }
        None

    }

    fn generate(&mut self, board: &Board, td: &ThreadData, filter: MoveFilter, next_stage: Stage) {
        self.idx = 0;
        self.moves = gen_moves(board, filter);
        self.moves.iter().for_each(|entry| MovePicker::score(entry, board, td, self.ply));
        self.stage = next_stage;
    }

    fn score(entry: &mut MoveListEntry, board: &Board, td: &ThreadData, ply: usize) {

        let mv = &entry.mv;
        if let (Some(attacker), Some(victim)) = (board.piece_at(mv.from()), board.captured(mv)) {
            // Score capture
            let victim_value = see::value(victim);
            let history_score = td.capture_history.get(board.stm, attacker, mv.to(), victim) as i32;
            entry.score = victim_value + history_score;
        } else if let Some(pc) = board.piece_at(mv.from()) {
            // Score quiet
            let quiet_score = td.quiet_history.get(board.stm, *mv) as i32;
            let mut cont_score = 0;
            for &prev_ply in &[1, 2] {
                if ply >= prev_ply  {
                    if let (Some(prev_mv), Some(prev_pc)) = (td.ss[ply - prev_ply].mv, td.ss[ply - prev_ply].pc) {
                        cont_score += td.cont_history.get(prev_mv, prev_pc, mv, pc) as i32;
                    }
                }
            }
            let is_killer = td.ss[ply].killer == Some(*mv);
            let base = if is_killer { 10000000 } else { 0 };
            entry.score = base + quiet_score + cont_score;
        }

    }

    fn pick(&mut self) -> Option<Move> {
        // Incremental selection sort
        loop {
            if self.moves.is_empty() || self.idx >= self.moves.len() {
                return None;
            }
            let mut best_index = self.idx;
            let mut best_score = self.moves.get(self.idx).map_or(0, |entry| entry.score);
            for j in self.idx + 1..self.moves.len() {
                if let Some(current) = self.moves.get(j) {
                    if current.score > best_score {
                        best_score = current.score;
                        best_index = j;
                    }
                } else {
                    break;
                }
            }
            if best_index != self.idx {
                self.moves.list.swap(self.idx, best_index);
            }

            if let Some(best_move) = self.moves.get(self.idx) {
                let mv = best_move.mv;
                if mv == self.tt_move {
                    self.idx += 1;
                    continue;
                }
                self.idx += 1;
                return Some(mv);
            }
            return None;
        }
    }

}