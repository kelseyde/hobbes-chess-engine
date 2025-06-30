use std::time::Duration;

pub struct SearchLimits {
    pub hard_time:  Option<Duration>,
    pub soft_time:  Option<Duration>,
    pub soft_nodes: Option<u64>,
    pub hard_nodes: Option<u64>,
    pub depth:      Option<u64>,
}

pub type FischerTime = (u64, u64); // (time, increment)

pub enum LimitType {
    Infinite,
    Movetime(u64),
    Fischer(u64, u64),
    Depth(u64),
    Nodes(u64),
}

impl SearchLimits {

    pub fn new(limit_type: LimitType) -> SearchLimits {
        match limit_type {
            LimitType::Infinite => Self::initInfinite(),
            LimitType::Movetime(time) => Self::initMovetime(time),
        }
    }

    pub fn initInfinite() -> SearchLimits {
        // self.limit_type = LimitType::Infinite;
        // self.hard_time = None;
        // self.soft_time = None;
        // self.hard_nodes = None;
        // self.soft_nodes = None;
        // self.depth = None;
    }

    pub fn initMovetime(time: u64) -> SearchLimits {
        SearchLimits {
            hard_time: Some(Duration::from_millis(time)),
            soft_time: None,
            nodes: None,
            depth: None,
        }
    }

    pub fn initFischer(time: u64, inc: u64) {

    }

}