use crate::evaluation::accumulator::psq::PieceSquareAccumulator;
use crate::evaluation::accumulator::threat::ThreatAccumulator;

pub mod psq;
pub mod threat;

#[derive(Default)]
pub struct Accumulator {
    pub psq: PieceSquareAccumulator,
    pub threat: ThreatAccumulator
}