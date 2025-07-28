use std::time::Instant;

use crate::correction::CorrectionHistories;
use crate::evaluation::network::NNUE;
use crate::history::Histories;
use crate::moves::Move;
use crate::search::{Score, SearchStack, MAX_PLY};
use crate::time::{LimitType, SearchLimits};
use crate::tt::TranspositionTable;

pub struct ThreadData {
    pub id: usize,
    pub main: bool,
    pub minimal_output: bool,
    pub tt: TranspositionTable,
    pub pv: PrincipalVariationTable,
    pub ss: SearchStack,
    pub nnue: NNUE,
    pub keys: Vec<u64>,
    pub root_ply: usize,
    pub history: Histories,
    pub correction_history: CorrectionHistories,
    pub lmr: LmrTable,
    pub node_table: NodeTable,
    pub limits: SearchLimits,
    pub start_time: Instant,
    pub nodes: u64,
    pub depth: i32,
    pub seldepth: usize,
    pub nmp_min_ply: i32,
    pub best_move: Move,
    pub best_score: i32,
}

impl Default for ThreadData {
    fn default() -> Self {
        ThreadData {
            id: 0,
            main: true,
            minimal_output: false,
            tt: TranspositionTable::new(64),
            pv: PrincipalVariationTable::default(),
            ss: SearchStack::new(),
            nnue: NNUE::default(),
            keys: Vec::new(),
            root_ply: 0,
            history: Histories::default(),
            correction_history: CorrectionHistories::default(),
            lmr: LmrTable::default(),
            node_table: NodeTable::new(),
            limits: SearchLimits::new(None, None, None, None, None),
            start_time: Instant::now(),
            nodes: 0,
            depth: 0,
            seldepth: 0,
            nmp_min_ply: 0,
            best_move: Move::NONE,
            best_score: Score::MIN,
        }
    }
}

impl ThreadData {

    pub fn reset(&mut self) {
        self.ss = SearchStack::new();
        self.node_table.clear();
        self.nodes = 0;
        self.depth = 1;
        self.seldepth = 0;
        self.best_move = Move::NONE;
        self.best_score = 0;
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.keys.clear();
        self.root_ply = 0;
        self.history.clear();
        self.correction_history.clear();
    }

    pub fn time(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }

    pub fn should_stop(&self, limit_type: LimitType) -> bool {
        if self.depth <= 1 {
            // Always clear the first depth, to ensure at least one legal move
            return false;
        }
        match limit_type {
            LimitType::Soft => self.soft_limit_reached(),
            LimitType::Hard => self.hard_limit_reached(),
        }
    }

    pub fn soft_limit_reached(&self) -> bool {
        let best_move_nodes = self.node_table.get(&self.best_move);

        if let Some(soft_time) = self.limits.scaled_soft_limit(self.depth, self.nodes, best_move_nodes) {
            if self.start_time.elapsed() >= soft_time {
                return true;
            }
        }

        if let Some(soft_nodes) = self.limits.soft_nodes {
            if self.nodes >= soft_nodes {
                return true;
            }
        }

        if let Some(depth_limit) = self.limits.depth {
            if self.depth >= depth_limit as i32 {
                return true;
            }
        }

        false
    }

    pub fn hard_limit_reached(&self) -> bool {
        if let Some(hard_time) = self.limits.hard_time {
            if self.start_time.elapsed() >= hard_time {
                return true;
            }
        }

        if let Some(hard_nodes) = self.limits.hard_nodes {
            if self.nodes >= hard_nodes {
                return true;
            }
        }

        if let Some(depth_limit) = self.limits.depth {
            if self.depth >= depth_limit as i32 {
                return true;
            }
        }

        false
    }
}

pub struct NodeTable {
    table: [[u64; 64]; 64],
}

impl NodeTable {

    pub const fn new() -> Self {
        NodeTable { table: [[0; 64]; 64] }
    }

    pub fn add(&mut self, mv: &Move, nodes: u64) {
        self.table[mv.from()][mv.to()] += nodes;
    }

    pub fn get(&self, mv: &Move) -> u64 {
        self.table[mv.from()][mv.to()]
    }

    pub const fn clear(&mut self) {
        *self = Self::new();
    }
}

pub struct PrincipalVariationTable {
    table: [[Move; MAX_PLY + 1]; MAX_PLY + 1],
    len: [usize; MAX_PLY + 1],
}

impl PrincipalVariationTable {
    pub const fn best_move(&self) -> Move {
        self.table[0][0]
    }

    pub fn line(&self) -> &[Move] {
        &self.table[0][..self.len[0]]
    }

    pub fn clear(&mut self, ply: usize) {
        self.len[ply] = 0;
    }

    pub fn update(&mut self, ply: usize, mv: Move) {
        self.table[ply][0] = mv;
        self.len[ply] = self.len[ply + 1] + 1;

        for i in 0..self.len[ply + 1] {
            self.table[ply][i + 1] = self.table[ply + 1][i];
        }
    }
}

impl Default for PrincipalVariationTable {
    fn default() -> Self {
        Self {
            table: [[Move::NONE; MAX_PLY + 1]; MAX_PLY + 1],
            len: [0; MAX_PLY + 1],
        }
    }
}

pub struct LmrTable {
    table: [[i32; 64]; 256],
}

impl LmrTable {
    pub fn reduction(&self, depth: i32, move_count: i32) -> i32 {
        self.table[depth.min(255) as usize][move_count.min(63) as usize]
    }
}

impl Default for LmrTable {
    fn default() -> Self {
        let base = 0.92;
        let divisor = 3.11;

        let mut table = [[0; 64]; 256];

        for depth in 1..256 {
            for move_count in 1..64 {
                let ln_depth = (depth as f32).ln();
                let ln_move_count = (move_count as f32).ln();
                let reduction = (base + (ln_depth * ln_move_count / divisor)) as i32;
                table[depth as usize][move_count as usize] = reduction;
            }
        }

        Self { table }
    }
}