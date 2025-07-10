use rand::Rng;

pub fn standard_gaussian() -> f64 {
    let mut rng = rand::thread_rng();
    let sum: f64 = (0..12).map(|_| rng.r#gen::<f64>()).sum();
    sum - 6.0
}

pub fn gaussian(mean: f64, std_dev: f64) -> f64 {
    mean + std_dev * standard_gaussian()
}

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
