use crate::board::side::Side;
use crate::evaluation::feature::Feature;

#[derive(Default, Clone, Copy)]
pub struct AccumulatorUpdate {
    pub add_count: usize,
    pub sub_count: usize,
    pub adds: [usize; 2],
    pub subs: [usize; 2],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AccumulatorUpdateType {
    None,
    Add,
    Sub,
    AddSub,
    AddSubSub,
    AddAddSubSub,
}

impl AccumulatorUpdate {
    pub fn push_add(&mut self, feature: Feature, perspective: Side, mirror: bool) {
        if self.add_count < 2 {
            self.adds[self.add_count] = feature.index(perspective, mirror);
            self.add_count += 1;
        }
    }

    pub fn push_sub(&mut self, feature: Feature, perspective: Side, mirror: bool) {
        if self.sub_count < 2 {
            self.subs[self.sub_count] = feature.index(perspective, mirror);
            self.sub_count += 1;
        }
    }

    pub fn update_type(&self) -> AccumulatorUpdateType {
        match (self.add_count, self.sub_count) {
            (0, 0) => AccumulatorUpdateType::None,
            (1, 0) => AccumulatorUpdateType::Add,
            (0, 1) => AccumulatorUpdateType::Sub,
            (1, 1) => AccumulatorUpdateType::AddSub,
            (1, 2) => AccumulatorUpdateType::AddSubSub,
            (2, 2) => AccumulatorUpdateType::AddAddSubSub,
            _ => AccumulatorUpdateType::None,
        }
    }
}