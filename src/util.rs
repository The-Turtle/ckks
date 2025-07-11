use rand::Rng;

/// Generates a random number from an approximate standard normal distribution (mean=0, std_dev=1).
///
/// This uses the central limit theorem approximation by summing 12 uniform random variables
/// and subtracting 6. This gives an approximate normal distribution with mean 0 and variance 1.
/// While not perfectly normal, it's computationally efficient and sufficient for many applications.
///
/// # Returns
/// A random floating-point number from the approximate standard normal distribution
pub fn standard_gaussian() -> f64 {
    let mut rng = rand::thread_rng();
    let sum: f64 = (0..12).map(|_| rng.r#gen::<f64>()).sum();
    sum - 6.0
}

/// Generates a random number from a normal distribution with specified mean and standard deviation.
///
/// # Arguments
/// * `mean` - The desired mean of the distribution
/// * `std_dev` - The desired standard deviation of the distribution
///
/// # Returns
/// A random floating-point number from the specified normal distribution
pub fn gaussian(mean: f64, std_dev: f64) -> f64 {
    mean + std_dev * standard_gaussian()
}

/// Computes the modular inverse of an odd number modulo 2^bits using Hensel's lemma.
///
/// This function finds x such that (a * x) ≡ 1 (mod 2^bits), where a is odd.
/// The algorithm uses Hensel's lemma to lift the inverse from smaller bit sizes.
///
/// # Arguments
/// * `a` - The number to find the inverse of (must be odd)
/// * `bits` - The number of bits defining the modulus (modulus = 2^bits)
///
/// # Returns
/// * `Some(inverse)` if the inverse exists (a is odd)
/// * `None` if no inverse exists (a is even)
///
/// # Algorithm
/// Uses Hensel's lemma to recursively compute the inverse:
/// 1. Base case: inverse of any odd number mod 2 is 1
/// 2. Recursive case: lift the inverse from mod 2^(bits-1) to mod 2^bits
///    by either keeping the current inverse or adding 2^(bits-1)
pub fn mod_power_of_2_inverse(a: usize, bits: usize) -> Option<usize> {
    if a % 2 == 0 {
        return None;
    }
    if bits == 1 {
        return Some(1);
    }
    let truncated_inverse = mod_power_of_2_inverse(a, bits - 1).unwrap();
    let mask = (1 << bits) - 1;
    if (truncated_inverse * a) & mask == 1 {
        Some(truncated_inverse)
    } else {
        Some(truncated_inverse | 1 << (bits - 1))
    }
}
