use crate::board::bitboard::Bitboard;
use crate::board::movegen::MoveFilter;
use crate::board::moves::{Move, MoveList, ScoredMove};
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

/// Selects the next move to try in a given position. We save time during search by trying moves in
/// stages. Typically, the TT move is tried first, then 'good' noisy moves, then quiet moves, and
/// finally 'bad' noisy moves. If we reach a beta cutoff in any of these stages, then we have saved
/// the time required to generate the moves in the later stages.
#[rustfmt::skip]
pub struct MovePicker {
    moves: MoveList,       // List of moves to pick from in the current stage
    pub stage: Stage,      // The current stage of move picking, e.g. generating quiets, trying captures.
    idx: usize,            // The index of the current move being searched.
    filter: MoveFilter,    // The filter to use when generating moves, e.g., by filtering out quiet moves in q-search.
    tt_move: Move,         // The move from the transposition table, which is tried first if it exists.
    ply: usize,            // The ply of the current search, used for history heuristics.
    threats: Bitboard,     // The squares threatened by the opponent, used for history heuristics.
    pub skip_quiets: bool, // Whether we should skip the remaining quiet moves.
    split_noisies: bool,   // Whether to split noisy moves into good and bad based on a SEE threshold.
    bad_noisies: MoveList, // Noisy moves that fail the SEE threshold, which are tried after quiet moves.
}

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

impl MovePicker {
    pub fn new(tt_move: Move, ply: usize, threats: Bitboard) -> Self {
        let stage = if tt_move.exists() {
            TTMove
        } else {
            GenerateNoisies
        };
        Self {
            moves: MoveList::new(),
            stage,
            idx: 0,
            filter: MoveFilter::Noisies,
            tt_move,
            ply,
            threats,
            skip_quiets: false,
            split_noisies: true,
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
            stage,
            idx: 0,
            filter,
            tt_move,
            ply,
            threats,
            skip_quiets: true,
            split_noisies: false,
            bad_noisies: MoveList::new(),
        }
    }

    /// Select the next move based on the current stage.
    pub fn next(&mut self, board: &Board, td: &ThreadData) -> Option<Move> {
        if self.stage == TTMove {
            self.stage = GenerateNoisies;
            if self.tt_move.exists() {
                return Some(self.tt_move);
            }
        }
        if self.stage == GenerateNoisies {
            self.idx = 0;
            self.gen_noisy_moves(board, td);
            self.stage = GoodNoisies;
        }
        if self.stage == GoodNoisies {
            if let Some(best_move) = self.pick_best(false) {
                return Some(best_move);
            } else {
                self.stage = GenerateQuiets;
            }
        }
        if self.stage == GenerateQuiets {
            self.idx = 0;
            if self.skip_quiets {
                self.stage = BadNoisies;
            } else {
                self.moves.list.clear();
                self.gen_quiet_moves(board, td);
                self.stage = Quiets;
            }
        }
        if self.stage == Quiets {
            if self.skip_quiets {
                self.idx = 0;
                self.stage = BadNoisies;
            } else if let Some(best_move) = self.pick_best(false) {
                return Some(best_move);
            } else {
                self.idx = 0;
                self.stage = BadNoisies;
            }
        }
        if self.stage == BadNoisies {
            if let Some(best_move) = self.pick_best(true) {
                return Some(best_move);
            } else {
                self.stage = Done;
            }
        }
        None
    }

    /// Generate all the quiet moves in the current position, and add them to the move list.
    #[inline(always)]
    fn gen_quiet_moves(&mut self, board: &Board, td: &ThreadData) {
        for entry in board.gen_moves(MoveFilter::Quiets).iter_mut() {
            if entry.mv == self.tt_move {
                continue;
            }
            score_move(entry, board, td, self.ply, self.threats);
            self.moves.add(*entry);
        }
    }

    /// Generate all the noisy moves in the current position using the provided filter, and add them
    /// to the 'good' or 'bad' noisy move list. Bad noisies are tried last.
    #[inline(always)]
    fn gen_noisy_moves(&mut self, board: &Board, td: &ThreadData) {
        for entry in board.gen_moves(self.filter).iter_mut() {
            if entry.mv == self.tt_move {
                continue;
            }
            score_move(entry, board, td, self.ply, self.threats);
            if is_good_noisy(entry, board, self.split_noisies) {
                self.moves.add(*entry);
            } else {
                self.bad_noisies.add(*entry);
            }
        }
    }

    /// Select the next move to try from the move list. We use an incremental selection sort, where
    /// only the best move is moved to the front each time. This is efficient since typically only
    /// a few moves will be tried during search, and so we save the time required to sort the rest.
    #[inline(always)]
    fn pick_best(&mut self, use_bad_noisies: bool) -> Option<Move> {
        let moves = if use_bad_noisies {
            &mut self.bad_noisies
        } else {
            &mut self.moves
        };
        if self.idx >= moves.len() {
            return None;
        }
        let packed = moves
            .list
            .iter()
            .enumerate()
            .skip(self.idx)
            .map(|(i, mv)| ((mv.score as i64) << 32) | ((u32::MAX as i64) - i as i64))
            .reduce(std::cmp::max)?;
        let best_index = (u32::MAX as usize) - (packed & 0xffffffff) as usize;

        if best_index != self.idx {
            moves.list.swap(self.idx, best_index);
        }
        let best_move = moves.list[self.idx].mv;
        self.idx += 1;
        Some(best_move)
    }
}

/// Assign a score to a move, determining the order in which moves are selected. Captures are scored
/// based on the value of the victim and the history score. Quiet moves are scored based on their
/// history scores, and given an additional bonus if they are a killer move.
#[inline(always)]
fn score_move(
    entry: &mut ScoredMove,
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
        let killer_bonus = if td.stack.is_killer(ply, *mv) {
            10_000_000
        } else {
            0
        };
        let direct_check_bonus = if board.gives_direct_check(*mv) && see::see(board, mv, -100, SeeType::Ordering) {
            8192
        } else {
            0
        };
        entry.score = quiet_score + cont_score + killer_bonus + direct_check_bonus;
    }
}

/// Determine whether the current noisy is 'good' or 'bad'. Queen and knight promos are always good.
/// Captures are sorted based on whether they pass a SEE threshold, which takes into account the
/// move's history score.
#[inline(always)]
fn is_good_noisy(entry: &ScoredMove, board: &Board, split_noisies: bool) -> bool {
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
            let threshold = -entry.score / movepick_see_divisor() + movepick_see_offset();
            match threshold {
                t if t > see::value(Queen, SeeType::Ordering) => false,
                t if t < -see::value(Queen, SeeType::Ordering) => true,
                _ => see(board, &entry.mv, threshold, SeeType::Ordering),
            }
        }
    }
}
