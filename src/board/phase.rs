#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Phase {
    P1 = 0,
    P2 = 1,
    P3 = 2,
}

impl Phase {
    pub fn from_usize(idx: usize) -> Self {
        match idx {
            0 => Phase::P1,
            1 => Phase::P2,
            _ => Phase::P3,
        }
    }
}
