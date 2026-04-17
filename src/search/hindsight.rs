use crate::board::Board;
use crate::search::parameters::{dynamic_policy_max, dynamic_policy_min, dynamic_policy_mult, hindsight_ext_eval_diff, hindsight_ext_min_depth, hindsight_ext_min_reduction, hindsight_red_eval_diff, hindsight_red_min_depth, hindsight_red_min_reduction};
use crate::search::score::is_defined;
use crate::search::thread::ThreadData;

#[inline]
pub fn should_extend(
    td: &ThreadData,
    root_node: bool,
    in_check: bool,
    singular_search: bool,
    static_eval: i32,
    depth: i32,
    ply: usize
) -> bool {
    !root_node
        && !in_check
        && !singular_search
        && depth >= hindsight_ext_min_depth()
        && td.stack[ply - 1].reduction >= hindsight_ext_min_reduction()
        && is_defined(td.stack[ply - 1].static_eval)
        && static_eval + td.stack[ply - 1].static_eval < hindsight_ext_eval_diff()
}

#[inline]
pub fn should_reduce(
    td: &ThreadData,
    root_node: bool,
    pv_node: bool,
    in_check: bool,
    singular_search: bool,
    static_eval: i32,
    depth: i32,
    ply: usize
) -> bool {
    !root_node
        && !pv_node
        && !in_check
        && !singular_search
        && depth >= hindsight_red_min_depth()
        && td.stack[ply - 1].reduction >= hindsight_red_min_reduction()
        && is_defined(td.stack[ply - 1].static_eval)
        && static_eval + td.stack[ply - 1].static_eval > hindsight_red_eval_diff()
}

#[inline]
pub fn should_update_history(
    td: &ThreadData,
    root_node: bool,
    in_check: bool,
    singular_search: bool,
    ply: usize
) -> bool {
    !in_check
        && !root_node
        && !singular_search
        && td.stack[ply - 1].mv.is_some()
        && td.stack[ply - 1].captured.is_none()
        && is_defined(td.stack[ply - 1].static_eval)
}

pub fn update_history(
    td: &mut ThreadData, 
    board: &Board, 
    ply: usize, 
    static_eval: i32
) {
    let prev_eval = td.stack[ply - 1].static_eval;
    let prev_mv = td.stack[ply - 1].mv.unwrap();
    let prev_pc = td.stack[ply - 1].pc.unwrap();
    let prev_threats = td.stack[ply - 1].threats;

    let value = dynamic_policy_mult() * -(static_eval + prev_eval);
    let bonus = value.clamp(dynamic_policy_min(), dynamic_policy_max()) as i16;
    td.history.quiet_history.update(!board.stm, &prev_mv, prev_pc, prev_threats, bonus, bonus);
}