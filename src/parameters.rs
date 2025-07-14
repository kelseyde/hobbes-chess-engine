use std::ops::Deref;

pub const ASP_DELTA: Tunable = Tunable::new("AspDelta", 24, 4, 36, 4);
pub const ASP_MIN_DEPTH: Tunable = Tunable::new("AspMinDepth", 4, 0, 8, 1);
pub const ASP_ALPHA_WIDENING_FACTOR: Tunable = Tunable::new("AspAlphaWideningFactor", 512, 256, 1024, 128);
pub const ASP_BETA_WIDENING_FACTOR: Tunable = Tunable::new("AspBetaWideningFactor", 512, 256, 1024, 128);

pub fn asp_delta() -> i32 {
    *ASP_DELTA
}

pub fn asp_min_depth() -> i32 {
    *ASP_MIN_DEPTH
}

pub fn asp_alpha_widening_factor() -> i32 {
    *ASP_ALPHA_WIDENING_FACTOR
}

pub fn asp_beta_widening_factor() -> i32 {
    *ASP_BETA_WIDENING_FACTOR
}

struct Tunable {
    name: &'static str,
    value: i32,
    min: i32,
    max: i32,
    step: i32
}

impl Tunable {

    pub const fn new(name: &'static str, value: i32, min: i32, max: i32, step: i32) -> Self {
        Self { name, value, min, max, step, }
    }

    pub fn print(&self) {
        println!("option name {} type spin default {} min {} max {}",
            self.name, self.value, self.min, self.max);
    }

}

impl Deref for Tunable {
    type Target = i32;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

#[cfg(feature = "spsa")]
pub fn print_tunables() {
    let tunables = [
        &ASP_DELTA,
        &ASP_MIN_DEPTH,
        &ASP_ALPHA_WIDENING_FACTOR,
        &ASP_BETA_WIDENING_FACTOR,
    ];
}

#[cfg(not(feature = "spsa"))]
pub fn print_tunables() {

}