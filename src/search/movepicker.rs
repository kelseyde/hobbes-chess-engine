use crate::board::bitboard::Bitboard;
use crate::board::movegen::MoveFilter;
use crate::board::moves::{Move, MoveList, MoveListEntry};
use crate::board::piece::Piece;
use crate::board::Board;
use crate::search::movepicker::Stage::{BadNoisies, Done, GoodNoisies};
use crate::search::parameters::{movepick_recapture_bonus, movepick_see_threshold};
use crate::search::see;
use crate::search::thread::ThreadData;
use Stage::{GenerateNoisies, GenerateQuiets, Quiets, TTMove};
use crate::search::stack::SearchStack;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    TTMove,
    GenerateNoisies,
    GoodNoisies,
    GenerateQuiets,
    Quiets,
    BadNoisies,
    Done,
}

pub struct MovePicker {
    moves: MoveList,
    filter: MoveFilter,
    idx: usize,
    pub stage: Stage,
    tt_move: Move,
    ply: usize,
    threats: Bitboard,
    pub skip_quiets: bool,
    see_threshold: Option<i32>,
    bad_noisies: MoveList,
}

impl MovePicker {
    pub fn new(tt_move: Move, ply: usize, threats: Bitboard) -> Self {
        let stage = if tt_move.exists() {
            TTMove
        } else {
            GenerateNoisies
        };
        Self {
            moves: MoveList::new(),
            filter: MoveFilter::Noisies,
            idx: 0,
            stage,
            tt_move,
            ply,
            threats,
            skip_quiets: false,
            see_threshold: Some(movepick_see_threshold()),
            bad_noisies: MoveList::new(),
        }
    }

    pub fn new_qsearch(tt_move: Move, filter: MoveFilter, ply: usize, threats: Bitboard) -> Self {
        let stage = if tt_move.exists() {
            TTMove
        } else {
            GenerateNoisies
        };
        Self {
            moves: MoveList::new(),
            filter,
            idx: 0,
            stage,
            tt_move,
            ply,
            threats,
            skip_quiets: true,
            see_threshold: None,
            bad_noisies: MoveList::new(),
        }
    }

    pub fn next(&mut self, board: &Board, td: &ThreadData, ss: &SearchStack) -> Option<Move> {
        if self.stage == TTMove {
            self.stage = GenerateNoisies;
            if self.tt_move.exists() && board.is_pseudo_legal(&self.tt_move) {
                return Some(self.tt_move);
            }
        }
        if self.stage == GenerateNoisies {
            self.idx = 0;
            let mut moves = board.gen_moves(self.filter);
            for entry in moves.iter() {
                MovePicker::score(entry, board, td, ss, self.ply, self.threats);

                let good_noisy = if entry.mv.is_promo() {
                    // Queen and knight promos are treated as good noisies
                    entry
                        .mv
                        .promo_piece()
                        .is_some_and(|p| p == Piece::Queen || p == Piece::Knight)
                } else {
                    // Captures are sorted based on whether they pass a SEE threshold
                    self.see_threshold
                        .map(|threshold| see(board, &entry.mv, threshold))
                        .unwrap_or(true)
                };

                if good_noisy {
                    self.moves.add(*entry);
                } else {
                    self.bad_noisies.add(*entry);
                }
            }
            self.stage = GoodNoisies;
        }
        if self.stage == GoodNoisies {
            if let Some(best_move) = self.pick(false) {
                return Some(best_move);
            } else {
                self.idx = 0;
                self.stage = GenerateQuiets;
            }
        }
        if self.stage == GenerateQuiets {
            self.idx = 0;
            if self.skip_quiets {
                self.stage = BadNoisies;
            } else {
                self.moves = board.gen_moves(MoveFilter::Quiets);
                self.moves
                    .iter()
                    .for_each(|entry| MovePicker::score(entry, board, td, ss, self.ply, self.threats));
                self.stage = Quiets;
            }
        }
        if self.stage == Quiets {
            if self.skip_quiets {
                self.idx = 0;
                self.stage = BadNoisies;
            } else if let Some(best_move) = self.pick(false) {
                return Some(best_move);
            } else {
                self.idx = 0;
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

    fn score(
        entry: &mut MoveListEntry,
        board: &Board,
        td: &ThreadData,
        ss: &SearchStack,
        ply: usize,
        threats: Bitboard,
    ) {
        let mv = &entry.mv;
        if let (Some(attacker), Some(victim)) = (board.piece_at(mv.from()), board.captured(mv)) {
            // Score capture
            let victim_value = see::value(victim);
            let history_score = td
                .history
                .capture_history_score(board, mv, attacker, victim);
            let is_recapture = ply > 0 && ss[ply - 1].mv.is_some_and(|prev_mv| {
                prev_mv.to() == mv.to()
            });
            let recapture_bonus = if is_recapture { movepick_recapture_bonus() } else { 0 };
            entry.score = victim_value + history_score + recapture_bonus;
        } else {
            // Score quiet
            let quiet_score = td
                .history
                .quiet_history_score(board, &td.ss, mv, ply, threats);
            let is_killer = td.ss[ply].killer == Some(*mv);
            let base = if is_killer { 10000000 } else { 0 };
            entry.score = base + quiet_score;
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
