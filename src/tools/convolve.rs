/// Generic convolution for 1, 2, and 3-way feature interactions.
///
/// - `features`: array of bools indicating which features are active
/// - `one_pair`: weights for single features
/// - `two_pair`: weights for feature pairs (length N*(N-1)/2)
/// - `three_pair`: weights for feature triplets (length N*(N-1)*(N-2)/6)
///
/// Returns the convolution sum as i32.
pub fn convolve<const N: usize>(
    features: &[bool; N],
    one_pair: &[i32; N],
    two_pair: &[i32],
    three_pair: &[i32],
) -> i32 {
    let mut output = 0;
    let mut two_index = 0;
    let mut three_index = 0;
    for i in 0..N {
        output += one_pair[i] * features[i] as i32;
        for j in (i + 1)..N {
            output += two_pair[two_index] * (features[i] && features[j]) as i32;
            for k in (j + 1)..N {
                output += three_pair[three_index] * (features[i] && features[j] && features[k]) as i32;
                three_index += 1;
            }
            two_index += 1;
        }
    }
    output
}
