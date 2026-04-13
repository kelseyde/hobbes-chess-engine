use crate::search::score;
use crate::search::tt::TTFlag;

/// A search window (alpha-beta bounds) passed to the recursive search functions.
///
/// `alpha` is the lower bound: we will not accept a score below this.
/// `beta`  is the upper bound: the opponent will not allow a score above this.
#[derive(Copy, Clone, Debug)]
pub struct Window {
    pub alpha: i32,
    pub beta: i32,
}

impl Window {
    /// Construct a window from explicit bounds.
    #[inline]
    pub const fn new(alpha: i32, beta: i32) -> Self {
        Self { alpha, beta }
    }

    /// Full-width window: the widest possible bounds.
    #[inline]
    pub const fn full() -> Self {
        Self::new(score::MIN, score::MAX)
    }

    /// Aspiration window centred on `score` with half-width `delta`.
    #[inline]
    pub fn aspiration(score: i32, delta: i32) -> Self {
        Self::new(score::clamp(score - delta), score::clamp(score + delta))
    }

    /// Null (zero) window just above `alpha`: `(alpha, alpha+1)`.
    /// Used for the most common PVS child searches.
    #[inline]
    pub const fn null(alpha: i32) -> Self {
        Self::new(alpha, alpha + 1)
    }

    /// Returns the window as seen by the child node: `(-beta, -alpha)`.
    #[inline]
    pub const fn flipped(self) -> Self {
        Self::new(-self.beta, -self.alpha)
    }

    /// Returns a null window just above the current `alpha`, then flipped for the child.
    /// Equivalent to passing `(-alpha-1, -alpha)` to the child.
    #[inline]
    pub const fn null_flipped(self) -> Self {
        Self::null(self.alpha).flipped()
    }

    /// Raise alpha to `score` (used when a move improves the lower bound).
    #[inline]
    pub fn raise_alpha(&mut self, score: i32) {
        if score > self.alpha {
            self.alpha = score;
        }
    }

    /// Returns whether `score` beats beta (causes a cut-off).
    #[inline]
    pub const fn fails_high(self, score: i32) -> bool {
        score >= self.beta
    }

    /// Returns whether `score` fails to beat alpha.
    #[inline]
    pub const fn fails_low(self, score: i32) -> bool {
        score <= self.alpha
    }

    /// Returns whether `score` is strictly inside the window.
    #[inline]
    pub const fn contains(self, score: i32) -> bool {
        score > self.alpha && score < self.beta
    }

    /// Derive the [`TTFlag`] for this score relative to the window.
    #[inline]
    pub const fn flag(self, score: i32) -> TTFlag {
        TTFlag::from_score(score, self.alpha, self.beta)
    }

    /// Widen the alpha side of an aspiration window after a fail-low.
    #[inline]
    pub fn widen_alpha(&mut self, score: i32, delta: i32) {
        self.beta = (self.alpha + self.beta) / 2;
        self.alpha = score::clamp(score - delta);
    }

    /// Widen the beta side of an aspiration window after a fail-high.
    #[inline]
    pub fn widen_beta(&mut self, score: i32, delta: i32) {
        self.beta = score::clamp(score + delta);
    }
}

