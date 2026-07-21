use crate::evaluation::accumulator::psq::PieceSquareAccumulator;
use crate::evaluation::accumulator::threat::ThreatAccumulator;

pub mod psq;
pub mod threat;

#[derive(Default)]
pub struct Accumulator {
    psq: PieceSquareAccumulator,
    threat: ThreatAccumulator
}

impl Accumulator {
    pub fn psq(&self) -> &PieceSquareAccumulator {
        &self.psq
    }

    pub fn threat(&self) -> &ThreatAccumulator {
        &self.threat
    }

    pub fn psq_mut(&mut self) -> &mut PieceSquareAccumulator {
        &mut self.psq
    }

    pub fn threat_mut(&mut self) -> &mut ThreatAccumulator {
        &mut self.threat
    }
}