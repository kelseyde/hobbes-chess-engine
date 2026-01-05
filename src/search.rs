pub mod correction;
pub mod history;
pub mod movepicker;
pub mod parameters;
pub mod score;
pub mod see;
pub mod stack;
pub mod thread;
pub mod time;
pub mod tt;

use crate::board::movegen::MoveFilter;
use crate::board::moves::{Move, MoveList};
use crate::board::{movegen, Board};
use crate::search::history::*;
use crate::search::movepicker::Stage::BadNoisies;
use crate::search::movepicker::{MovePicker, Stage};
use crate::search::score::{format_score, Score};
use crate::search::see::see;
use crate::search::thread::ThreadData;
use crate::search::time::LimitType::{Hard, Soft};
use crate::search::tt::TTFlag;
use crate::search::tt::TTFlag::Upper;
use arrayvec::ArrayVec;
use parameters::*;
use crate::board::piece::Piece;
use crate::board::piece::Piece::King;

pub const MAX_PLY: usize = 256;

/// Classical alpha-beta search with iterative deepening.
/// Alpha-beta search seeks to reduce the number of nodes that need to be evaluated in the search
/// tree. It does this by pruning branches that are guaranteed to be worse than the best move found
/// so far, or that are guaranteed to be 'too good' and could only be reached by sup-optimal play
/// by the opponent.
pub fn search(board: &Board, td: &mut ThreadData) -> (Move, i32) {
    td.pv.clear(0);
    td.nnue.activate(board);

    let root_moves = board.gen_legal_moves();
    match root_moves.len {
        0 => return handle_no_legal_moves(board, td),
        1 => return handle_one_legal_move(board, td, &root_moves),
        _ => {}
    }

    let mut alpha = Score::MIN;
    let mut beta = Score::MAX;
    let mut score = 0;
    let mut delta = asp_delta();
    let mut reduction = 0;

    // Iterative Deepening
    // Search the position to a fixed depth, increasing the depth each iteration until the maximum
    // depth is reached or the search is aborted.
    while td.depth < MAX_PLY as i32 && !td.should_stop(Soft) {
        // Aspiration Windows
        // Use the score from the previous iteration to guess the score from the current iteration.
        // Based on this guess, we narrow the alpha-beta window around the previous score, causing
        // more cut-offs and thus speeding up the search. If the true score is outside the window,
        // a costly re-search is required.
        if td.depth >= asp_min_depth() {
            alpha = (score - delta).max(Score::MIN);
            beta = (score + delta).min(Score::MAX);
        }

        loop {
            let search_depth = td.depth - reduction;
            score = alpha_beta(board, td, search_depth, 0, alpha, beta, false);

            if td.main && !td.minimal_output {
                print_search_info(td);
            }

            if td.should_stop(Hard) || Score::is_mate(score) {
                break;
            }

            // Adjust the aspiration window in case the score fell outside the current window.
            match score {
                s if s <= alpha => {
                    beta = (alpha + beta) / 2;
                    alpha = (score - delta).max(Score::MIN);
                    delta += (delta * 100) / asp_alpha_widening_factor();
                    reduction = 0;
                }
                s if s >= beta => {
                    beta = (score + delta).min(Score::MAX);
                    delta += (delta * 100) / asp_beta_widening_factor();
                    reduction = (reduction + 1).min(3);
                }
                _ => break,
            }
        }

        if td.should_stop(Hard) || Score::is_mate(score) {
            break;
        }

        delta = asp_delta() + score * score / asp_prev_score_div();
        reduction = 0;
        td.depth += 1;
    }

    // Print the final search stats
    if td.main {
        print_search_info(td);
    }

    // If time expired before a best move was found in search, pick the first legal move.
    if !td.best_move.exists() {
        if let Some(root_move) = root_moves.get(0) {
            println!("info error no best move was found in search, returning random move");
            td.best_move = root_move.mv;
        }
    }

    (td.best_move, td.best_score)
}

#[rustfmt::skip]
fn alpha_beta(board: &Board,
              td: &mut ThreadData,
              mut depth: i32,
              ply: usize,
              mut alpha: i32,
              mut beta: i32,
              cut_node: bool) -> i32 {

    // If search is aborted, exit immediately
    if td.should_stop(Hard) {
        return alpha;
    }

    // A PV (principal variation) node is one that falls within the alpha-beta window.
    let pv_node = beta - alpha > 1;

    // The root node is the first node in the search tree, and is thus also always a PV node.
    let root_node = ply == 0;

    // Determine if we are currently in check.
    let threats = board.threats;
    let in_check = threats.contains(board.king_sq(board.stm));
    td.ss[ply].threats = threats;

    // Update the selective search depth
    if ply + 1 > td.seldepth {
        td.seldepth = ply + 1;
    }

    // If depth is reached, drop into quiescence search
    if depth <= 0 && !in_check {
        return qs(board, td, alpha, beta, ply);
    }

    // Ensure depth is not negative
    if depth < 0 {
        depth = 0;
    }

    // If drawn by repetition, insufficient material or fifty move rule, return a draw score.
    if ply > 0 && is_draw(td, board) {
        return Score::DRAW;
    }

    // If the maximum depth is reached, return the static evaluation of the position
    if ply >= MAX_PLY {
        return td.nnue.evaluate(board);
    }

    // Mate Distance Pruning
    // If we have already found a mate, prune nodes where no shorter mate is possible
    alpha = alpha.max(Score::mated_in(ply));
    beta = beta.min(Score::mate_in(ply));
    if alpha >= beta {
        return alpha;
    }

    // Clear the principal variation for this ply.
    if pv_node {
        td.pv.clear(ply);
    }

    let singular = td.ss[ply].singular;
    let singular_search = singular.is_some();

    let mut tt_hit = false;
    let mut tt_move = Move::NONE;
    let mut tt_move_noisy = false;
    let mut tt_score = Score::MIN;
    let mut tt_eval = Score::MIN;
    let mut has_tt_score = false;
    let mut tt_flag = TTFlag::Lower;
    let mut tt_depth = 0;
    let mut tt_pv = pv_node;

    // Transposition table
    // Check if this node has already been searched before. If it has, and the depth + bounds match
    // the requirements of the current search, then we can directly return the score from the TT.
    // If the depth and bounds do not match, we can still use information from the TT - such as the
    // best move, score, and static eval - to inform the current search.
    if !singular_search {
        if let Some(entry) = td.tt.probe(board.hash()) {
            tt_hit = true;
            tt_score = entry.score(ply) as i32;
            tt_eval = entry.static_eval() as i32;
            has_tt_score = Score::is_defined(tt_score);
            tt_depth = entry.depth() as i32;
            tt_flag = entry.flag();
            tt_pv = tt_pv || entry.pv();
            if can_use_tt_move(board, &entry.best_move()) {
                tt_move = entry.best_move();
                tt_move_noisy = board.is_noisy(&tt_move)
            }

            if !root_node
                && !pv_node
                && tt_depth >= depth
                && entry.flag().bounds_match(tt_score, alpha, beta) {
                return tt_score;
            }
        }
    }

    // Static Evaluation
    // Obtain a static evaluation of the current board state. In leaf nodes, this is the final score
    // used in search. In non-leaf nodes, it is used as a guide for several heuristics, such as
    // extensions, reductions and pruning.
    let mut raw_eval = Score::MIN;
    let mut static_eval = Score::MIN;

    if !in_check {
        raw_eval = if singular_search {
            td.ss[ply].raw_eval
        } else if tt_hit && Score::is_defined(tt_eval) {
            tt_eval
        } else {
            td.nnue.evaluate(board)
        };
        if !tt_hit {
            td.tt.insert(board.hash(), Move::NONE, 0, raw_eval, depth, ply, TTFlag::None, tt_pv);
        }
        let correction = td.correction_history.correction(board, &td.ss, ply);
        static_eval = raw_eval + correction;
    }

    td.ss[ply].raw_eval = raw_eval;
    td.ss[ply].static_eval = static_eval;

    // We are 'improving' if the static eval of the current position is greater than it was on our
    // previous turn. If improving, we can be more aggressive in our beta pruning - where the eval
    // is too high - but should be more cautious in our alpha pruning - where the eval is too low.
    let improvement = calc_improvement(td, ply, static_eval, in_check);
    let improving = improvement > 0;

    // Hindsight history updates
    // Use the difference between the static eval in the current node and parent node to update the
    // history score for the parent move.
    if !in_check
        && !root_node
        && !singular_search
        && td.ss[ply - 1].mv.is_some()
        && td.ss[ply - 1].captured.is_none()
        && Score::is_defined(td.ss[ply - 1].static_eval) {

        let prev_eval = td.ss[ply - 1].static_eval;
        let prev_mv = td.ss[ply - 1].mv.unwrap();
        let prev_threats = td.ss[ply - 1].threats;

        let value = dynamic_policy_mult() * -(static_eval + prev_eval);
        let bonus = value.clamp(dynamic_policy_min(), dynamic_policy_max()) as i16;
        td.history.quiet_history.update(!board.stm, &prev_mv, prev_threats, bonus);
    }

    // Hindsight extension
    // If we reduced depth in the parent node, but now the static eval indicates the position is
    // improving, we correct the reduction 'in hindsight' by extending depth in the current node.
    if !root_node
        && !in_check
        && !singular_search
        && depth >= hindsight_ext_min_depth()
        && td.ss[ply - 1].reduction >= hindsight_ext_min_reduction()
        && Score::is_defined(td.ss[ply - 1].static_eval)
        && static_eval + td.ss[ply - 1].static_eval < hindsight_ext_eval_diff() {
        depth += 1;
    }

    // Hindsight reduction
    // If we reduced depth in the parent node, and now the static eval confirms the position is
    // improving, we affirm the parent node's reduction 'in hindsight' by reducing even further.
    if !root_node
        && !pv_node
        && !in_check
        && !singular_search
        && depth >= hindsight_red_min_depth()
        && td.ss[ply - 1].reduction >= hindsight_red_min_reduction()
        && Score::is_defined(td.ss[ply - 1].static_eval)
        && static_eval + td.ss[ply - 1].static_eval > hindsight_red_eval_diff() {
        depth -= 1;
    }

    // Pre-move-loop pruning: If the static eval indicates a fail-high or fail-low, there are several
    // heuristics we can employ to prune the node and its entire subtree, without searching any moves.
    if !root_node && !pv_node && !in_check && !singular_search{

        // Reverse Futility Pruning
        // Skip nodes where the static eval is far above beta and will thus likely fail high.
        let futility_margin = rfp_base()
            + rfp_scale() * depth
            - rfp_improving_scale() * improving as i32
            - rfp_tt_move_noisy_scale() * tt_move_noisy as i32;
        if depth <= rfp_max_depth()
            && static_eval - futility_margin >= beta
            && tt_flag != Upper {
            return beta + (static_eval - beta) / 3;
        }

        // Razoring
        // Drop into q-search for nodes where the eval is far below alpha, and will likely fail low.
        if !pv_node && static_eval < alpha - razor_base() - razor_scale() * depth * depth {
            return qs(board, td, alpha, beta, ply);
        }

        // Null Move Pruning
        // Skip nodes where giving the opponent an extra move (making a 'null move') still fails high.
        if depth >= nmp_min_depth()
            && static_eval >= beta + nmp_margin()
            && ply as i32 > td.nmp_min_ply
            && board.has_non_pawns()
            && tt_flag != Upper {

            let r = nmp_base_reduction()
                + depth / nmp_depth_divisor()
                + ((static_eval - beta) / nmp_eval_divisor()).min(nmp_eval_max_reduction())
                + tt_move_noisy as i32;

            let mut board = *board;
            board.make_null_move();
            td.nodes += 1;
            td.keys.push(board.hash());
            let score = -alpha_beta(&board, td, depth - r, ply + 1, -beta, -beta + 1, !cut_node);
            td.keys.pop();

            if score >= beta {
                // At low depths, we can directly return the result of the null move search.
                if td.nmp_min_ply > 0 || depth <= 14 {
                    return if Score::is_mate(score) { beta } else {score };
                }

                // At high depths, we do a normal search to verify the null move result.
                td.nmp_min_ply = (3 * (depth - r) / 4) + ply as i32;
                let verif_score = alpha_beta(&board, td, depth - r, ply, beta - 1, beta, true);
                td.nmp_min_ply = 0;

                if verif_score >= beta {
                    return score;
                }
            }
        }

    }

    // Internal Iterative Reductions
    // If the position has not been searched yet, the search will be potentially expensive. So we
    // search with a reduced depth expecting to record a move that we can later re-use.
    if !root_node
        && depth >= iir_min_depth()
        && (pv_node || cut_node)
        && (!tt_hit || tt_move.is_null() || tt_depth < depth - iir_tt_depth_offset()) {
        depth -= 1;
    }

    // We have decided that the current node should not be pruned and is worth examining further.
    // Now we begin iterating through the moves in the position and searching deeper in the tree.

    let mut move_picker = MovePicker::new(tt_move, ply, threats);

    let mut legal_moves = 0;
    let mut searched_moves = 0;
    let mut quiet_count = 0;
    let mut capture_count = 0;
    let mut best_score = Score::MIN;
    let mut best_move = Move::NONE;
    let mut flag = TTFlag::Upper;

    let mut quiets = ArrayVec::<Move, 32>::new();
    let mut captures = ArrayVec::<Move, 32>::new();

    while let Some(mv) = move_picker.next(board, td) {

        if !board.is_legal(&mv) {
            continue;
        }

        legal_moves += 1;

        if singular.is_some_and(|s| s == mv) {
            continue;
        }

        let pc = board.piece_at(mv.from()).unwrap();
        let captured = board.captured(&mv);
        let is_quiet = captured.is_none();
        let is_mate_score = Score::is_mate(best_score);
        let history_score = td.history.history_score(board, &td.ss, &mv, ply, threats, pc, captured);
        let base_reduction = td.lmr.reduction(depth, legal_moves);
        let lmr_depth = depth.saturating_sub(base_reduction);

        let mut extension = 0;

        // Check Extensions
        // If we are in check then the position is likely tactical, so we extend the search depth.
        if in_check {
            extension = 1;
        }

        // Futility Pruning
        // Skip quiet moves when the static evaluation + some margin is still below alpha.
        let futility_margin = fp_base()
            + fp_scale() * lmr_depth
            - legal_moves * fp_movecount_mult()
            + history_score / fp_history_divisor();
        if !root_node
            && !in_check
            && is_quiet
            && lmr_depth < fp_max_depth()
            && !is_mate_score
            && static_eval + futility_margin <= alpha {
            move_picker.skip_quiets = true;
            continue;
        }

        // Late Move Pruning
        // Skip quiet moves ordered very late in the list.
        if !pv_node
            && !root_node
            && !is_mate_score
            && is_quiet
            && depth <= lmp_max_depth()
            && searched_moves > late_move_threshold(depth, improving) {
            move_picker.skip_quiets = true;
            continue;
        }

        // History Pruning
        // Skip quiet moves that have a bad history score.
        if !pv_node
            && !root_node
            && !is_mate_score
            && is_quiet
            && depth <= hp_max_depth()
            && history_score < hp_scale() * depth * depth {
            move_picker.skip_quiets = true;
            continue
        }

        // Bad Noisy Pruning
        // Skip bad noisies when the static evaluation + some margin is still below alpha.
        let futility_margin = static_eval + bnp_scale() * lmr_depth;
        if !pv_node
            && !in_check
            && lmr_depth < bnp_max_depth()
            && move_picker.stage == Stage::BadNoisies
            && futility_margin <= alpha {
            break;
        }

        // SEE Pruning
        // Skip moves that lose material once all the pieces have been exchanged.
        let see_threshold = if is_quiet {
            lmr_depth * pvs_see_quiet_scale()
                - history_score / pvs_see_quiet_history_div()
                - tt_pv as i32 * lmr_depth * pvs_see_quiet_ttpv_scale()
        } else {
            depth * depth * pvs_see_noisy_scale()
                - history_score / pvs_see_noisy_history_div()
        };
        if !pv_node
            && depth <= pvs_see_max_depth()
            && threats.contains(mv.to())
            && searched_moves >= 1
            && !Score::is_mate(best_score)
            && !see(board, &mv, see_threshold) {
            continue;
        }

        // Singular Extensions
        // Do a reduced-depth search with the TT move excluded. If the result of that search plus
        // some margin doesn't beat the TT score, we assume the TT move is 'singular' (i.e. the
        // only good move), and extend the search depth.
        if !root_node
            && !singular_search
            && tt_hit
            && mv == tt_move
            && depth >= se_min_depth()
            && tt_flag != TTFlag::Upper
            && tt_depth >= depth - se_tt_depth_offset() {

            let s_beta_mult = depth * (1 + (tt_pv && !pv_node) as i32);
            let s_beta = (tt_score - s_beta_mult * se_beta_scale() / 16).max(-Score::MATE + 1);
            let s_depth = (depth - se_depth_offset()) / se_depth_divisor();

            td.ss[ply].singular = Some(mv);
            let score = alpha_beta(board, td, s_depth, ply, s_beta - 1, s_beta, cut_node);
            td.ss[ply].singular = None;

            if score < s_beta {
                extension = 1;
                extension += (!pv_node && score < s_beta - se_double_ext_margin()) as i32;
            } else if s_beta >= beta {
                return (s_beta * s_depth + beta) / (s_depth + 1);
            } else if tt_score >= beta {
                extension = -3;
            } else if cut_node {
                extension = -2;
            } else if tt_score <= alpha {
                extension = -1;
            }

        }

        // We have decided that the current move should not be pruned and is worth searching further.
        // Therefore, we make the move on the board and search the resulting position.
        let original_board = board;
        let mut board = *board;
        td.nnue.update(&mv, pc, captured, &board);
        board.make(&mv);

        td.ss[ply].mv = Some(mv);
        td.ss[ply].pc = Some(pc);
        td.ss[ply].captured = captured;
        td.keys.push(board.hash());
        td.tt.prefetch(board.hash());

        searched_moves += 1;
        td.nodes += 1;

        let initial_nodes = td.nodes;
        let mut new_depth = depth - 1 + extension;

        let mut score = Score::MIN;

        // Principal Variation Search
        // We assume that the first move will be best, and search all others with a null window and/or
        // reduced depth. If any of those moves beat alpha, we re-search with a full window and depth.
        if depth >= lmr_min_depth() && searched_moves > lmr_min_moves() + root_node as i32 + 2 * pv_node as i32 {

            // Late Move Reductions
            // Moves ordered late in the list are less likely to be good, so we reduce the depth.
            let mut r = base_reduction * 1024;
            r -= lmr_ttpv_base() * tt_pv as i32;
            r -= lmr_ttpv_tt_score() * (tt_pv && has_tt_score && tt_score > alpha) as i32;
            r -= lmr_ttpv_tt_depth() * (tt_pv && has_tt_score && tt_depth >= depth) as i32;
            r += lmr_cut_node() * cut_node as i32;
            r -= lmr_capture() * captured.is_some() as i32;
            r += lmr_improving() * !improving as i32;
            r -= lmr_shallow() * (depth == lmr_min_depth()) as i32;
            r -= lmr_killer() * td.ss[ply].killer.is_some_and(|k| k == mv) as i32;
            r -= extension * 1024 / lmr_extension_divisor();
            r -= is_quiet as i32 * ((history_score - lmr_hist_offset()) / lmr_hist_divisor()) * 1024;
            r -= !is_quiet as i32 * captured.map_or(0, |c| see::value(c) / lmr_mvv_divisor());
            r += (is_quiet && !see::see(original_board, &mv, 0)) as i32 * lmr_quiet_see();
            let reduced_depth = (new_depth - (r / 1024)).clamp(1, new_depth);

            // For moves eligible for reduction, we apply the reduction and search with a null window.
            td.ss[ply].reduction = r;
            score = -alpha_beta(&board, td, reduced_depth, ply + 1, -alpha - 1, -alpha, true);
            td.ss[ply].reduction = 0;

            // If the reduced search beat alpha, re-search at full depth, with a null window.
            if score > alpha && new_depth > reduced_depth {
                // Adjust the depth of the re-search based on the score from the reduced search.
                let do_deeper_margin = best_score + lmr_deeper_base() + lmr_deeper_scale() * depth / lmr_deeper_div();
                let do_shallower_margin = best_score + new_depth;
                new_depth += (score > do_deeper_margin) as i32;
                new_depth -= (score < do_shallower_margin) as i32;

                if new_depth > reduced_depth {
                    score = -alpha_beta(&board, td, new_depth, ply + 1, -alpha - 1, -alpha, !cut_node);

                    if is_quiet && (score <= alpha || score >= beta) {
                        let bonus = lmr_conthist_bonus(depth, score >= beta);
                        td.history.update_continuation_history(&td.ss, ply, &mv, pc, bonus);
                    }
                }
            }
        }
        // If we're skipping late move reductions - either due to being in a PV node, or searching
        // the first move, or another reason - then we search at full depth with a null-window.
        else if !pv_node || searched_moves > 1 {
            score = -alpha_beta(&board, td, new_depth, ply + 1, -alpha - 1, -alpha, !cut_node);
        }

        // If we're in a PV node and searching the first move, or the score from reduced search beat
        // alpha, then we search with full depth and alpha-beta window.
        if pv_node && (searched_moves == 1 || score > alpha) {
            score = -alpha_beta(&board, td, new_depth, ply + 1, -beta, -alpha, false);
        }

        // Register the current move, to update its history score later
        if is_quiet && quiet_count < 32 {
            quiets.push(mv);
            quiet_count += 1;
        } else if captured.is_some() && capture_count < 32 {
            captures.push(mv);
            capture_count += 1;
        }

        td.ss[ply].mv = None;
        td.ss[ply].pc = None;
        td.ss[ply].captured = None;
        td.keys.pop();
        td.nnue.undo();

        if root_node {
            td.node_table.add(&mv, td.nodes - initial_nodes);
        }

        if td.should_stop(Hard) {
            break;
        }

        if score > best_score {
            best_score = score;
        }

        // If the score is greater than alpha, then this is the best move we have examined so far.
        // We therefore update alpha to the current score and update best move to the current move.
        if score > alpha {
            alpha = score;
            best_move = mv;
            flag = TTFlag::Exact;

            if pv_node {
                td.pv.update(ply, mv);
                if root_node {
                    td.best_move = mv;
                    td.best_score = score;
                }
            }

            // If the score is greater than beta, then this position is 'too good' - our opponent
            // won't let us get here assuming perfect play. There is therefore no point searching
            // further, and we can cut off the search.
            if score >= beta {
                flag = TTFlag::Lower;
                break;
            }

            // Alpha-raise reduction
            // It is unlikely that multiple moves raise alpha, therefore, if we have already raised
            // alpha, we can reduce the search depth for the remaining moves.
            if depth > alpha_raise_min_depth()
                && depth < alpha_raise_max_depth()
                && !is_mate_score {
                depth -= 1;
            }

        }
    }

    // Update history tables
    // When the best move causes a beta cut-off, we update the history tables to reward the best move
    // and punish the other searched moves. Doing so will improve move ordering in subsequent searches.
    if best_move.exists() {
        let pc = board.piece_at(best_move.from()).unwrap();
        let new_tt_move = tt_move.exists() && best_move != tt_move;

        let quiet_bonus = quiet_history_bonus(depth)
            - cut_node as i16 * quiet_hist_cutnode_offset() as i16
            + new_tt_move as i16 * quiet_hist_ttmove_bonus() as i16
            + capture_count as i16 * quiet_hist_capture_mult() as i16;

        let quiet_malus = quiet_history_malus(depth)
            + new_tt_move as i16 * quiet_hist_ttmove_malus() as i16;

        let capt_bonus = capture_history_bonus(depth)
            + new_tt_move as i16 * capt_hist_ttmove_bonus() as i16;

        let capt_malus = capture_history_malus(depth)
            + new_tt_move as i16 * capt_hist_ttmove_malus() as i16;

        let cont_bonus = cont_history_bonus(depth)
            - cut_node as i16 * cont_hist_cutnode_offset() as i16
            + new_tt_move as i16 * cont_hist_ttmove_bonus() as i16
            + capture_count as i16 * cont_hist_capture_mult() as i16;

        let cont_malus = cont_history_malus(depth)
            + new_tt_move as i16 * cont_hist_ttmove_malus() as i16;

        let pc_bonus = piece_history_bonus(depth);
        let pc_malus = piece_history_malus(depth);

        if pc != King {
            td.history.piece_history.update(board.stm, pc, pc_bonus);
            // for weaker_pc in Piece::weaker_pieces(pc) {
            //     td.history.piece_history.update(board.stm, *weaker_pc, -pc_malus);
            // }
            td.debug_stats.insert(String::from("PawnScore"), td.history.piece_history.get(board.stm, Piece::Pawn) as i64);
            td.debug_stats.insert(String::from("KnightScore"), td.history.piece_history.get(board.stm, Piece::Knight) as i64);
            td.debug_stats.insert(String::from("BishopScore"), td.history.piece_history.get(board.stm, Piece::Bishop) as i64);
            td.debug_stats.insert(String::from("RookScore"), td.history.piece_history.get(board.stm, Piece::Rook) as i64);
            td.debug_stats.insert(String::from("QueenScore"), td.history.piece_history.get(board.stm, Piece::Queen) as i64);
        }
        if let Some(captured) = board.captured(&best_move) {
             // If the best move was a capture, give it a capture history bonus.
            td.history.capture_history.update(board.stm, pc, best_move.to(), captured, capt_bonus);
        } else {
            // If the best move was quiet, record it as a 'killer' and give it a quiet history bonus.
            td.ss[ply].killer = Some(best_move);
            td.history.quiet_history.update(board.stm, &best_move, threats, quiet_bonus);
            td.history.update_continuation_history(&td.ss, ply, &best_move, pc, cont_bonus);

            // Penalise all the other quiets which failed to cause a beta cut-off.
            for mv in quiets.iter() {
                if mv != &best_move {
                    td.history.quiet_history.update(board.stm, mv, threats, quiet_malus);
                    td.history.update_continuation_history(&td.ss, ply, mv, pc, cont_malus);
                }
            }
        }

        // Regardless of whether the best move was quiet or a capture, penalise all other captures.
        for mv in captures.iter() {
            if mv != &best_move {
                if let Some(captured) = board.captured(mv) {
                    td.history.capture_history.update(board.stm, pc, mv.to(), captured, capt_malus);
                }
            }
        }
    }

    // Prior Countermove Bonus
    // The current node failed low, meaning the parent node will fail high. If the parent move is
    // quiet it will receive a quiet history bonus - but we give it one here too, which ensures the
    // best move is updated also during PVS re-searches, hopefully leading to better move ordering.
    if !root_node
        && flag == TTFlag::Upper
        && td.ss[ply - 1].captured.is_none() {
        if let Some(prev_mv) = td.ss[ply - 1].mv {
            let prev_threats = td.ss[ply - 1].threats;
            let quiet_bonus = prior_countermove_bonus(depth);
            td.history.quiet_history.update(!board.stm, &prev_mv, prev_threats, quiet_bonus);
        }
    }

    // Checkmate / Stalemate Detection
    if legal_moves == 0 {
        return if singular_search {
            alpha
        } else if in_check {
            -Score::MATE + ply as i32
        } else {
            Score::DRAW
        };
    }

    // Update static eval correction history.
    if !in_check
        && !singular_search
        && flag.bounds_match(best_score, static_eval, static_eval)
        && (!best_move.exists() || !board.is_noisy(&best_move)) {
        td.correction_history.update_correction_history(board, &td.ss, depth, ply, static_eval, best_score);
    }

    // Store the best move and score in the transposition table
    if !singular_search && !td.hard_limit_reached(){
        td.tt.insert(board.hash(), best_move, best_score, raw_eval, depth, ply, flag, tt_pv);
    }

    best_score
}

/// Quiescence Search.
/// Extend the search by searching captures until a quiet position is reached, where there are no
/// more captures and therefore limited potential for winning tactics that drastically alter the
/// evaluation. Used to mitigate the 'horizon effect'.
fn qs(board: &Board, td: &mut ThreadData, mut alpha: i32, beta: i32, ply: usize) -> i32 {
    let pv_node = beta - alpha > 1;

    // If search is aborted, exit immediately
    if td.should_stop(Hard) {
        return alpha;
    }

    // Update the selective search depth
    if ply + 1 > td.seldepth {
        td.seldepth = ply + 1;
    }

    // If drawn by repetition, insufficient material or fifty move rule, return zero.
    if ply > 0 && is_draw(td, board) {
        return Score::DRAW;
    }

    // Clear the principal variation for this ply.
    if pv_node {
        td.pv.clear(ply);
    }

    // If the maximum depth is reached, return the static evaluation of the position.
    if ply >= MAX_PLY {
        return td.nnue.evaluate(board);
    }

    // Determine if we are currently in check.
    let threats = board.threats;
    let in_check = threats.contains(board.king_sq(board.stm));
    td.ss[ply].threats = threats;

    // Transposition Table Lookup
    let mut tt_hit = false;
    let mut tt_pv = pv_node;
    let mut tt_move = Move::NONE;
    let mut tt_eval = Score::MIN;
    if let Some(entry) = td.tt.probe(board.hash()) {
        tt_hit = true;
        tt_pv = tt_pv || entry.pv();
        tt_eval = entry.static_eval() as i32;
        if can_use_tt_move(board, &entry.best_move()) {
            tt_move = entry.best_move();
        }
        let score = entry.score(ply) as i32;

        if entry.flag().bounds_match(score, alpha, beta) {
            return score;
        }
    }

    let mut raw_eval = Score::MIN;
    let mut static_eval = Score::MIN;

    if !in_check {
        raw_eval = if tt_hit && Score::is_defined(tt_eval) {
            tt_eval
        } else {
            td.nnue.evaluate(board)
        };
        if !tt_hit {
            td.tt.insert(board.hash(), Move::NONE, 0, raw_eval, 0, ply, TTFlag::None, tt_pv);
        }
        let correction = td.correction_history.correction(board, &td.ss, ply);
        static_eval = raw_eval + correction;
    }

    if !in_check {
        // If we are not in check, then we have the option to 'stand pat', i.e. decline to continue
        // the capture chain, if the static evaluation of the position is good enough.
        if static_eval > alpha {
            alpha = static_eval
        }
        if alpha >= beta {
            return alpha;
        }
    }

    let filter = if in_check {
        MoveFilter::All
    } else {
        MoveFilter::Captures
    };
    let mut move_picker = MovePicker::new_qsearch(tt_move, filter, ply, threats);

    let mut move_count = 0;

    let futility_margin = static_eval + qs_futility_threshold();

    let mut best_score = static_eval;
    let mut best_move = Move::NONE;
    let mut flag = TTFlag::Upper;

    while let Some(mv) = move_picker.next(board, td) {
        if !board.is_legal(&mv) {
            continue;
        }

        if move_picker.stage == BadNoisies {
            break;
        }

        let pc = board.piece_at(mv.from()).unwrap();
        let captured = board.captured(&mv);
        let is_quiet = captured.is_none();
        let is_recapture = board.recapture_sq.is_some_and(|sq| sq == mv.to());
        let is_mate_score = Score::is_mate(best_score);

        // Late Move Pruning
        if !in_check && !is_recapture && !is_mate_score && move_count >= 2 {
            break;
        }

        // Futility Pruning
        // Skip captures that don't win material when the static eval is far below alpha.
        if !in_check && !is_mate_score && futility_margin <= alpha && !see::see(board, &mv, 1) {
            if best_score < futility_margin {
                best_score = futility_margin;
            }
            continue;
        }

        // SEE Pruning
        // Skip moves which lose material once all the pieces are swapped off.
        if !in_check && !see::see(board, &mv, qs_see_threshold()) {
            continue;
        }

        // Evasion Pruning
        // In check, stop searching quiet moves after finding at least one non-losing move.
        if in_check && move_count > 1 && is_quiet && !is_mate_score {
            break;
        }

        let mut board = *board;
        td.nnue.update(&mv, pc, captured, &board);

        board.make(&mv);
        td.ss[ply].mv = Some(mv);
        td.ss[ply].pc = Some(pc);
        td.ss[ply].captured = captured;
        td.keys.push(board.hash());
        td.tt.prefetch(board.hash());

        move_count += 1;
        td.nodes += 1;

        let score = -qs(&board, td, -beta, -alpha, ply + 1);

        td.ss[ply].mv = None;
        td.ss[ply].pc = None;
        td.ss[ply].captured = None;
        td.keys.pop();
        td.nnue.undo();

        if td.should_stop(Hard) {
            break;
        }

        if score > best_score {
            best_score = score;
        }

        if score > alpha {
            alpha = score;
            best_move = mv;
            flag = TTFlag::Exact;

            if pv_node {
                td.pv.update(ply, mv);
            }

            if score >= beta {
                flag = TTFlag::Lower;
                break;
            }
        }
    }

    if move_count == 0 && in_check {
        return -Score::MATE + ply as i32;
    }

    if best_score >= beta && Score::is_defined(best_score) {
        best_score = (best_score + beta) / 2;
    }

    // Write to transposition table
    if !td.hard_limit_reached() {
        td.tt.insert(
            board.hash(),
            best_move,
            best_score,
            raw_eval,
            0,
            ply,
            flag,
            tt_pv,
        );
    }

    best_score
}

fn is_draw(td: &ThreadData, board: &Board) -> bool {
    board.is_fifty_move_rule() || board.is_insufficient_material() || is_repetition(board, td)
}

fn is_repetition(board: &Board, td: &ThreadData) -> bool {
    let curr_hash = board.hash();
    let mut repetitions = 0;
    let end = td.keys.len().saturating_sub(board.hm as usize + 1);
    for ply in (end..td.keys.len().saturating_sub(2)).rev() {
        let hash = td.keys[ply];
        repetitions += u8::from(curr_hash == hash);

        // Two-fold repetition of positions within the search tree
        if repetitions == 1 && ply >= td.root_ply {
            return true;
        }

        // Three-fold repetition including positions before search root
        if repetitions == 2 {
            return true;
        }
    }
    false
}

fn can_use_tt_move(board: &Board, tt_move: &Move) -> bool {
    tt_move.exists() && board.is_pseudo_legal(tt_move) && board.is_legal(tt_move)
}

fn calc_improvement(td: &ThreadData, ply: usize, static_eval: i32, in_check: bool) -> i32 {
    if ply >= 2 && Score::is_defined(td.ss[ply - 2].static_eval) && !in_check {
        static_eval - td.ss[ply - 2].static_eval
    } else if ply >= 4 && Score::is_defined(td.ss[ply - 4].static_eval) && !in_check {
        static_eval - td.ss[ply - 4].static_eval
    } else {
        0
    }
}

fn late_move_threshold(depth: i32, improving: bool) -> i32 {
    let base = if improving {
        lmp_improving_base()
    } else {
        lmp_base()
    };
    let scale = if improving {
        lmp_improving_scale()
    } else {
        lmp_scale()
    };
    (base + depth * scale) / 10
}

fn print_search_info(td: &mut ThreadData) {
    let depth = td.depth;
    let seldepth = td.seldepth;
    let best_score = format_score(td.best_score);
    let nodes = td.nodes;
    let time = td.start_time.elapsed().as_millis();
    let nps = if time > 0 && nodes > 0 {
        (nodes as u128 / time) * 1000
    } else {
        0
    };
    let hashfull = td.tt.fill();
    print!(
        "info depth {} seldepth {} score {} nodes {} time {} nps {} hashfull {} pv",
        depth, seldepth, best_score, nodes, time, nps, hashfull
    );

    // TODO fix illegal PV moves
    // for mv in td.pv.line() {
    //     print!(" {}", mv.to_uci());
    // }
    //
    // if td.pv.line().is_empty() {
    //     print!(" {}", td.pv.best_move().to_uci());
    // }
    print!(" {}", td.best_move.to_uci());
    println!();
}

fn handle_one_legal_move(board: &Board, td: &mut ThreadData, root_moves: &MoveList) -> (Move, i32) {
    let mv = root_moves.get(0).unwrap().mv;
    let static_eval = td.nnue.evaluate(board);
    td.depth = 1;
    td.best_move = mv;
    td.best_score = static_eval;
    print_search_info(td);
    (td.best_move, td.best_score)
}

fn handle_no_legal_moves(board: &Board, td: &mut ThreadData) -> (Move, i32) {
    println!("info error no legal moves");
    let in_check = movegen::is_check(board, board.stm);
    let score = if in_check { -Score::MATE } else { Score::DRAW };
    td.best_move = Move::NONE;
    td.best_score = score;
    (td.best_move, td.best_score)
}
