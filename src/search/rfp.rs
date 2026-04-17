use crate::search::parameters::{rfp_base, rfp_improving_scale, rfp_max_depth, rfp_scale, rfp_tt_move_noisy_scale};
use crate::search::tt::TTFlag;
use crate::search::tt::TTFlag::Upper;

#[inline]
#[clippy::allow(clippy::too_many_arguments)]
pub fn is_futile(
    root_node: bool,
    pv_node: bool,
    in_check: bool,
    singular_search: bool,
    improving: bool,
    tt_move_noisy: bool,
    tt_flag: TTFlag,
    depth: i32,
    static_eval: i32,
    beta: i32,
) -> bool {
    !root_node
    && !pv_node
    && !in_check
    && !singular_search
    && depth <= rfp_max_depth()
    && tt_flag != Upper
    && static_eval - futility_margin(depth, improving, tt_move_noisy) >= beta
}

fn futility_margin(depth: i32, improving: bool, tt_move_noisy: bool) -> i32 {
    rfp_base()
        + rfp_scale() * depth
        - rfp_improving_scale() * improving as i32
        - rfp_tt_move_noisy_scale() * tt_move_noisy as i32
}

#[inline]
pub fn futility_value(static_eval: i32, beta: i32) -> i32 {
    beta + (static_eval - beta) / 3
}