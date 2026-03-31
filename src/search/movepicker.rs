use crate::board::bitboard::Bitboard;
use crate::board::movegen::MoveFilter;
use crate::board::moves::{Move, MoveList, MoveListEntry};
use crate::board::piece::Piece;
use crate::board::piece::Piece::Queen;
use crate::board::Board;
use crate::search::movepicker::Stage::{BadNoisies, Done, GoodNoisies};
use crate::search::parameters::{movepick_see_divisor, movepick_see_offset};
use crate::search::see;
use crate::search::see::SeeType;
use crate::search::thread::ThreadData;
use Piece::Knight;
use Stage::{GenerateNoisies, GenerateQuiets, Quiets, TTMove};

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
    filter: MoveFilter,
    idx: usize,
    pub stage: Stage,
    tt_move: Move,
    ply: usize,
    threats: Bitboard,
    pub skip_quiets: bool,
    split_noisies: bool,
    good_noisies: MoveList,
    bad_noisies: MoveList,
    quiets: MoveList,
}

impl MovePicker {
    pub fn new(tt_move: Move, ply: usize, threats: Bitboard) -> Self {
        let stage = if tt_move.exists() {
            TTMove
        } else {
            GenerateNoisies
        };
        Self {
            filter: MoveFilter::Noisies,
            idx: 0,
            stage,
            tt_move,
            ply,
            threats,
            skip_quiets: false,
            split_noisies: true,
            good_noisies: MoveList::new(),
            bad_noisies: MoveList::new(),
            quiets: MoveList::new(),
        }
    }

    pub fn new_qsearch(tt_move: Move, filter: MoveFilter, ply: usize, threats: Bitboard) -> Self {
        let stage = if tt_move.exists() {
            TTMove
        } else {
            GenerateNoisies
        };
        Self {
            filter,
            idx: 0,
            stage,
            tt_move,
            ply,
            threats,
            skip_quiets: true,
            split_noisies: false,
            good_noisies: MoveList::new(),
            bad_noisies: MoveList::new(),
            quiets: MoveList::new(),
        }
    }

    pub fn next(&mut self, board: &Board, td: &ThreadData) -> Option<Move> {
        if self.stage == TTMove {
            self.stage = GenerateNoisies;
            if self.tt_move.exists() && board.is_pseudo_legal(&self.tt_move) {
                return Some(self.tt_move);
            }
        }
        if self.stage == GenerateNoisies {
            self.idx = 0;
            let mut moves: MoveList = board.gen_moves(self.filter);
            for entry in moves.iter().filter(|entry| entry.mv != self.tt_move) {
                MovePicker::score(entry, board, td, self.ply, self.threats);
                let good_noisy = is_good_noisy(entry, board, self.split_noisies);
                let moves = if good_noisy {
                    &mut self.good_noisies
                } else {
                    &mut self.bad_noisies
                };
                moves.add(*entry);
            }
            self.stage = GoodNoisies;
        }
        if self.stage == GoodNoisies {
            if self.idx != self.good_noisies.list.len() {
                return Some(self.pick_best(self.stage))
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
                let mut moves: MoveList = board.gen_moves(MoveFilter::Quiets);
                for entry in moves.iter().filter(|entry| entry.mv != self.tt_move) {
                    MovePicker::score(entry, board, td, self.ply, self.threats);
                    self.quiets.add(*entry);
                }
                self.stage = Quiets;
            }
        }
        if self.stage == Quiets {
            if self.skip_quiets {
                self.idx = 0;
                self.stage = BadNoisies;
            } else if self.idx != self.quiets.list.len() {
                return Some(self.pick_best(self.stage));
            } else {
                self.idx = 0;
                self.stage = BadNoisies;
            }
        }
        if self.stage == BadNoisies && self.idx != self.bad_noisies.list.len() {
            return Some(self.pick_best(self.stage))
        } else {
            self.stage = Done;
        }
        None
    }

    fn score(
        entry: &mut MoveListEntry,
        board: &Board,
        td: &ThreadData,
        ply: usize,
        threats: Bitboard,
    ) {
        let mv = &entry.mv;
        if let (Some(attacker), Some(victim)) = (board.piece_at(mv.from()), board.captured(mv)) {
            // Score capture
            let victim_value = see::value(victim, SeeType::Ordering);
            let history_score = td
                .history
                .capture_history_score(board, mv, attacker, victim);
            entry.score = 16 * victim_value + history_score;
        } else if let Some(pc) = board.piece_at(mv.from()) {
            // Score quiet
            let quiet_score = td.history.quiet_history_score(board, mv, pc, threats);
            let cont_score = td.history.cont_history_score(board, &td.stack, mv, ply);
            let is_killer = td.stack[ply].killer == Some(*mv);
            let base = if is_killer { 10000000 } else { 0 };
            entry.score = base + quiet_score + cont_score;
        }
    }

    fn pick_best(&mut self, stage: Stage) -> Move {
        let moves = match stage {
            GoodNoisies => &mut self.good_noisies,
            Quiets => &mut self.quiets,
            BadNoisies => &mut self.bad_noisies,
            _ => unreachable!(),
        };
        let packed = moves
            .list
            .iter()
            .enumerate()
            .skip(self.idx)
            .map(|(i, mv)| (i as i32) | (mv.score) << 16)
            .fold(i32::MIN, std::cmp::max);
        let idx = packed as usize & 0xffff;
        moves.list.swap(self.idx, idx);
        self.idx += 1;
        moves.list[idx].mv
    }

}

fn is_good_noisy(entry: &MoveListEntry, board: &Board, split_noisies: bool) -> bool {
    if entry.mv.is_promo() {
        // Queen and knight promos are treated as good noisies
        entry
            .mv
            .promo_piece()
            .is_some_and(|p| p == Queen || p == Knight)
    } else {
        // Captures are sorted based on whether they pass a SEE threshold
        if !split_noisies {
            true
        } else {
            let threshold =
                -entry.score / movepick_see_divisor() + movepick_see_offset();
            match threshold {
                t if t > see::value(Queen, SeeType::Ordering) => false,
                t if t < -see::value(Queen, SeeType::Ordering) => true,
                _ => see(board, &entry.mv, threshold, SeeType::Ordering),
            }
        }
    }
}
