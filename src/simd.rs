#[cfg(feature = "avx512")]
pub(crate) mod avx2 {

    pub fn forward(us: &[i16; HIDDEN], them: &[i16; HIDDEN], weights: &[i16; HIDDEN * 2]) -> i32 {
        println!("Not implemented");
        0
    }

}

#[cfg(feature = "avx512")]
pub(crate) mod avx512 {

    pub fn forward(us: &[i16; HIDDEN], them: &[i16; HIDDEN], weights: &[i16; HIDDEN * 2]) -> i32 {
        println!("Not implemented");
        0
    }

}

#[cfg(all(not(target_feature = "avx2"), not(feature = "avx512")))]
pub(crate) mod scalar {
    use crate::network::{HIDDEN, QA};

    pub fn forward(us: &[i16; HIDDEN], them: &[i16; HIDDEN], weights: &[i16; HIDDEN * 2]) -> i32 {
        let mut output = 0;

        for (&input, &weight) in us.iter().zip(weights[..HIDDEN].iter()) {
            let clipped = input.clamp(0, QA as i16);
            let result = clipped * weight;
            output += result as i32 * clipped as i32;
        }

        for (&input, &weight) in them.iter().zip(weights[HIDDEN..].iter()) {
            let clipped = input.clamp(0, QA as i16);
            let result = clipped * weight;
            output += result as i32 * clipped as i32;
        }

        output
    }

}