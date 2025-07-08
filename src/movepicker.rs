use crate::board::Board;
use crate::moves::{Move, MoveList, MoveListEntry};
use crate::thread::ThreadData;
use crate::{movegen, see};
use movegen::{gen_moves, MoveFilter};
use Stage::{GenerateNoisies, GenerateQuiets, Quiets, TTMove};
use crate::movepicker::Stage::{BadNoisies, Done, GoodNoisies};
use crate::see::see;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    TTMove,
    GenerateNoisies,
    GoodNoisies,
    GenerateQuiets,
    Quiets,
    BadNoisies,
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
    see_threshold: Option<i32>,
    bad_noisies: MoveList,
}

impl MovePicker {

    pub fn new(tt_move: Move, ply: usize) -> Self {
        let stage = if tt_move.exists() { TTMove } else { GenerateNoisies };
        Self {
            moves: MoveList::new(),
            filter: MoveFilter::Noisies,
            idx: 0,
            stage,
            tt_move,
            ply,
            skip_quiets: false,
            see_threshold: Some(0),
            bad_noisies: MoveList::new(),
        }
    }

    pub fn new_qsearch(tt_move: Move, filter: MoveFilter, ply: usize) -> Self {
        let stage = if tt_move.exists() { TTMove } else { GenerateNoisies };
        Self {
            moves: MoveList::new(),
            filter,
            idx: 0,
            stage,
            tt_move,
            ply,
            skip_quiets: true,
            see_threshold: None,
            bad_noisies: MoveList::new(),
        }
    }

    pub fn next(&mut self, board: &Board, td: &ThreadData) -> Option<Move> {

        if self.stage == TTMove {
            self.stage = GenerateNoisies;
            if self.tt_move.exists() {
                return Some(self.tt_move);
            }
        }
        if self.stage == GenerateNoisies {
            self.idx = 0;
            let mut moves = gen_moves(board, self.filter);
            for entry in moves.iter() {
                MovePicker::score(entry, board, td, self.ply);
                if self.see_threshold
                    .map(|threshold| !see(board, &entry.mv, threshold))
                    .unwrap_or(false) {
                    self.bad_noisies.add(*entry);
                } else {
                    self.moves.add(*entry);
                }
            }
            self.stage = GoodNoisies;
        }
        if self.stage == GoodNoisies {
            if let Some(best_move) = self.pick(false) {
                return Some(best_move)
            } else {
                self.idx = 0;
                self.stage = GenerateQuiets;
            }
        }
        if self.stage == GenerateQuiets {
            if self.skip_quiets {
                self.idx = 0;
                self.stage = BadNoisies;
            }
            self.idx = 0;
            self.moves = gen_moves(board, MoveFilter::Quiets);
            self.moves.iter().for_each(|entry| MovePicker::score(entry, board, td, self.ply));
            self.stage = Quiets;
        }
        if self.stage == Quiets {
            if self.skip_quiets {
                self.idx = 0;
                self.stage = BadNoisies;
            }
            if let Some(best_move) = self.pick(false) {
                return Some(best_move)
            } else {
                self.stage = BadNoisies;
            }
        }
        if self.stage == BadNoisies {
            if let Some(best_move) = self.pick(true) {
                return Some(best_move);
            } else {
                self.stage = Done;
            }
        }
        None

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

    fn pick(&mut self, use_bad_noisies: bool) -> Option<Move> {
        let moves = if use_bad_noisies {
            &mut self.bad_noisies
        } else {
            &mut self.moves
        };
        loop {
            if moves.is_empty() || self.idx >= moves.len() {
                return None;
            }
            let mut best_index = self.idx;
            let mut best_score = moves.get(self.idx).map_or(0, |entry| entry.score);
            for j in self.idx + 1..moves.len() {
                if let Some(current) = moves.get(j) {
                    if current.score > best_score {
                        best_score = current.score;
                        best_index = j;
                    }
                } else {
                    break;
                }
            }
            if best_index != self.idx {
                moves.list.swap(self.idx, best_index);
            }

            if let Some(best_move) = moves.get(self.idx) {
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