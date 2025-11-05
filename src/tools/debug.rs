// Stats and percentile estimation utilities.
// Ported from provided Zig DebugStats implementation.

use rand::prelude::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashMap;

pub const PERCENTILES: [f64; 9] = [0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9];
const NUM_PERCENTILES: usize = PERCENTILES.len();

#[derive(Clone, Debug, Default)]
pub struct DebugStatsMap {
    pub map: HashMap<String, DebugStats>,
}

#[derive(Clone, Debug)]
pub struct DebugStats {
    pub sum: i64,
    pub sum_sqr: u128,
    pub count: i64,
    pub min: i64,
    pub max: i64,
    pub percentiles: [f64; NUM_PERCENTILES],
    rng: StdRng,
}

impl Default for DebugStats {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugStats {
    pub fn new() -> Self {
        Self {
            sum: 0,
            sum_sqr: 0,
            count: 0,
            min: i64::MAX,
            max: i64::MIN,
            percentiles: [0.0; NUM_PERCENTILES],
            rng: StdRng::seed_from_u64(0),
        }
    }

    pub fn add(&mut self, data_point: i64) {
        self.sum += data_point;
        self.sum_sqr += data_point.abs() as u128 * data_point.abs() as u128;
        self.count += 1;
        if data_point < self.min {
            self.min = data_point;
        }
        if data_point > self.max {
            self.max = data_point;
        }

        let data_point_f = data_point as f64;
        let step = 0.001 * (1.0 + self.standard_deviation());
        let r: f64 = self.rng.random();

        for (i, percentile) in PERCENTILES.iter().enumerate() {
            let p = percentile / 100.0;
            let val = &mut self.percentiles[i];
            if data_point_f > *val && r > 1.0 - p {
                *val += step;
            } else if data_point_f < *val && r > p {
                *val -= step;
            }
        }
    }

    pub fn median(&self) -> f64 {
        for (i, percentile) in PERCENTILES.iter().enumerate() {
            if (*percentile - 50.0).abs() < f64::EPSILON {
                return self.percentiles[i];
            }
        }
        panic!("50th percentile definition missing for median");
    }

    pub fn skewness(&self) -> f64 {
        let sd = self.standard_deviation();
        if sd == 0.0 {
            return 0.0;
        }
        (self.average() - self.median()) / sd
    }

    pub fn average(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum as f64 / self.count as f64
    }

    pub fn variance(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let sum = self.sum as f64;
        let sum_sqr = self.sum_sqr as f64;
        let count = self.count as f64;
        (sum_sqr - sum * sum / count) / count
    }

    pub fn standard_deviation(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn print(&self) {
        println!("Count: {}", self.count);
        println!("Sum: {}", self.sum);
        println!("Min: {}", self.min);
        println!("Max: {}", self.max);
        println!("Average: {:.2}", self.average());
        println!("Standard Deviation: {:.2}", self.standard_deviation());
        println!("Skewness: {:.2}", self.skewness());
        for (i, percentile) in PERCENTILES.iter().enumerate() {
            println!("Percentile {:.1}%: {:.2}", percentile, self.percentiles[i]);
        }
    }
}

impl DebugStatsMap {
    pub fn insert(&mut self, key: String, data_point: i64) {
        let stats = self.map.entry(key).or_insert_with(DebugStats::new);
        stats.add(data_point);
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn print(&self) {
        for (key, stats) in &self.map {
            println!("{}:", key);
            stats.print();
            println!();
        }
    }
}