

pub struct SearchLimits {

}

pub enum LimitType {
    Infinite,
    Movetime(u64),
    Fischer(u64, u64),
    Depth(u64),
    SoftNodes(u64),
    HardNodes(u64)
}

impl LimitType {



}