//! Fast Fourier Transform (FFT) implementation for complex numbers and convolution operations.
//! This module provides efficient FFT, inverse FFT, and convolution functions for both 1D and 2D data.

use num_complex::Complex;
use once_cell::sync::Lazy;
use std::f64::consts::PI;

/// Maximum number of bits for cached roots of unity (2^11 = 2048 values)
const ROOTS_CACHE_SIZE: usize = 11;

/// Tau constant (2π) for angle calculations
const TAU: f64 = 2.0 * PI;

/// Pre-computed roots of unity for efficient FFT computation.
/// These are the complex numbers e^(2πik/n) for k = 0, 1, ..., n-1 where n = 2^ROOTS_CACHE_SIZE.
/// Cached to avoid recomputation during FFT operations.
static ROOTS_OF_UNITY: Lazy<Vec<Complex<f64>>> = Lazy::new(|| {
    let n = 1 << ROOTS_CACHE_SIZE;
    (0..n)
        .map(|k| {
            let theta = TAU * (k as f64) / (n as f64);
            Complex::from_polar(1.0, theta)
        })
        .collect()
});

/// Of the 2^bit 2^bit-th roots of unity, returns the k^th one,
/// which equals e^(2πik/2^bits). Uses pre-computed values from
/// the ROOTS_OF_UNITY cache for efficiency.
///
/// # Arguments
/// * `k` - The index of the root of unity
/// * `bits` - Log2 of the FFT size (must be <= ROOTS_CACHE_SIZE)
///
/// # Returns
/// The complex number e^(2πik/2^bits)
pub fn root_of_unity(k: i64, bits: usize) -> Complex<f64> {
    debug_assert!(bits <= ROOTS_CACHE_SIZE);
    let mask = (1 << bits) - 1;
    let i = (k & mask) as usize;
    ROOTS_OF_UNITY[i << (ROOTS_CACHE_SIZE - bits)]
}

/// Core FFT/IFFT implementation using Cooley-Tukey divide-and-conquer algorithm.
///
/// # Arguments
/// * `input` - Input vector of complex numbers (length must be power of 2)
/// * `is_inverse` - If true, performs inverse FFT; if false, performs forward FFT
///
/// # Returns
/// Transformed vector of complex numbers
fn transform(input: Vec<Complex<f64>>, is_inverse: bool) -> Vec<Complex<f64>> {
    let n = input.len();
    debug_assert!(n.is_power_of_two());

    let n_exp = n.trailing_zeros() as usize;

    if n_exp == 0 {
        let value = input[0].clone();
        return vec![value];
    }

    let evens: Vec<Complex<f64>> = input.iter().step_by(2).cloned().collect();
    let odds: Vec<Complex<f64>> = input.iter().skip(1).step_by(2).cloned().collect();
    let even_fft = transform(evens, is_inverse);
    let odd_fft = transform(odds, is_inverse);

    let mut output = Vec::with_capacity(1 << n_exp);
    let half_n = 1 << (n_exp - 1);
    let sign = if is_inverse { -1 } else { 1 };

    for k in 0..half_n {
        let even = even_fft[k];
        let odd = odd_fft[k];
        output.push(even + root_of_unity(k as i64 * sign, n_exp) * odd);
    }
    for k in 0..half_n {
        let even = even_fft[k];
        let odd = odd_fft[k];
        output.push(even - root_of_unity(k as i64 * sign, n_exp) * odd);
    }

    output
}

/// Computes the Fast Fourier Transform of the input vector.
///
/// # Arguments
/// * `input` - Vector of complex numbers (length must be power of 2)
///
/// # Returns
/// FFT of the input vector
pub fn fft(input: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    transform(input, false)
}

/// Computes the Inverse Fast Fourier Transform of the input vector.
///
/// # Arguments
/// * `input` - Vector of complex numbers (length must be power of 2)
///
/// # Returns
/// Inverse FFT of the input vector
pub fn ifft(input: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    transform(input, true)
}

/// Computes the convolution of two complex vectors using FFT.
/// Uses the convolution theorem: convolution in time domain = multiplication
/// in frequency domain.
/// The input vectors must have lengths whose sum is at most 2^ROOTS_CACHE_SIZE+1.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
///
/// # Returns
/// Convolution result a * b
fn convolve_complex(mut a: Vec<Complex<f64>>, mut b: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let n = (a.len() + b.len() - 1).next_power_of_two();
    debug_assert!(n <= 1 << ROOTS_CACHE_SIZE, "Input vectors are too long");
    a.resize(n, Complex::new(0.0, 0.0));
    b.resize(n, Complex::new(0.0, 0.0));
    let fft_a = fft(a);
    let fft_b = fft(b);
    let mut fft_c = Vec::with_capacity(n);
    for i in 0..n {
        fft_c.push(fft_a[i] * fft_b[i]);
    }
    let mut c = ifft(fft_c);
    for i in 0..n {
        c[i] /= n as f64;
    }
    c
}

/// Computes the convolution of two integer vectors using FFT.
/// Converts integers to complex numbers, performs complex convolution, then converts back.
/// If the inputs are too long or too large, the convolution may have rounding issues.
///
/// # Arguments
/// * `a` - First input vector of integers
/// * `b` - Second input vector of integers
///
/// # Returns
/// Convolution result as a vector of integers
pub fn convolve_int(a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
    let a_as_complex = a
        .into_iter()
        .map(|x| Complex::new(*x as f64, 0.0))
        .collect();
    let b_as_complex = b
        .into_iter()
        .map(|x| Complex::new(*x as f64, 0.0))
        .collect();
    let c = convolve_complex(a_as_complex, b_as_complex);
    c.into_iter().map(|x| x.re.round() as u64).collect()
}

/// 2D FFT/IFFT implementation using row-column decomposition.
/// Applies 1D transforms to each row, then to each column of the result.
///
/// # Arguments
/// * `input` - 2D matrix of complex numbers (dimensions must be powers of 2)
/// * `is_inverse` - If true, performs inverse 2D FFT; if false, performs forward 2D FFT
///
/// # Returns
/// 2D transformed matrix
fn transform_2d(input: Vec<Vec<Complex<f64>>>, is_inverse: bool) -> Vec<Vec<Complex<f64>>> {
    let n = input.len();
    debug_assert!(n.is_power_of_two(), "Row count must be a power of two");
    let m = input[0].len();
    debug_assert!(m.is_power_of_two(), "Column count must be a power of two");
    debug_assert!(
        input.iter().all(|row| row.len() == m),
        "All rows must have the same length"
    );

    let row_transformed: Vec<Vec<Complex<f64>>> = input
        .into_iter()
        .map(|row| transform(row, is_inverse))
        .collect();

    let mut transposed = vec![vec![Complex::new(0.0, 0.0); n]; m];
    for i in 0..n {
        for j in 0..m {
            transposed[j][i] = row_transformed[i][j];
        }
    }

    transposed
        .into_iter()
        .map(|row| transform(row, is_inverse))
        .collect()
}

/// Computes the 2D Fourier Transform of the input matrix.
fn fft_2d(input: Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    transform_2d(input, false)
}

/// Computes the 2D Inverse Fourier Transform of the input matrix.
fn ifft_2d(input: Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    transform_2d(input, true)
}

/// Computes the 2D convolution of two complex matrices using 2D FFT.
/// Extends the convolution theorem to 2D: 2D convolution = element-wise
/// multiplication in 2D frequency domain. The input matrices row counts
/// whose sum is at most 2^ROOTS_CACHE_SIZE+1, and the maximum sum of column
/// counts must also be at most 2^ROOTS_CACHE_SIZE+1.
///
/// # Arguments
/// * `a` - First input matrix
/// * `b` - Second input matrix
///
/// # Returns
/// 2D convolution result
fn convolve_complex_2d(
    mut a: Vec<Vec<Complex<f64>>>,
    mut b: Vec<Vec<Complex<f64>>>,
) -> Vec<Vec<Complex<f64>>> {
    let a_row_count = a.len();
    let b_row_count = b.len();
    let a_col_count = a.iter().map(|row| row.len()).max().unwrap_or(0);
    let b_col_count = b.iter().map(|row| row.len()).max().unwrap_or(0);
    let n = (a_row_count + b_row_count - 1).next_power_of_two();
    let m = (a_col_count + b_col_count - 1).next_power_of_two();
    debug_assert!(n <= 2 << ROOTS_CACHE_SIZE, "Input matrices are too large");
    debug_assert!(m <= 2 << ROOTS_CACHE_SIZE, "Input matrices are too large");

    a.resize(n, vec![Complex::new(0.0, 0.0); m]);
    b.resize(n, vec![Complex::new(0.0, 0.0); m]);
    for i in 0..n {
        a[i].resize(m, Complex::new(0.0, 0.0));
        b[i].resize(m, Complex::new(0.0, 0.0));
    }

    let a_transformed = fft_2d(a);
    let b_transformed = fft_2d(b);

    let mut c_transformed = vec![vec![Complex::new(0.0, 0.0); n]; m];
    for i in 0..m {
        for j in 0..n {
            c_transformed[i][j] = a_transformed[i][j] * b_transformed[i][j];
        }
    }
    let mut scaled_output = ifft_2d(c_transformed);
    for i in 0..n {
        for j in 0..m {
            scaled_output[i][j] /= (m * n) as f64;
        }
    }
    scaled_output
}

/// Computes the 2D convolution of two integer matrices using 2D FFT.
/// Converts integers to complex numbers, performs complex 2D convolution, then converts back.
/// There might be rounding issues due to precision loss if the input matrices
/// are large.
///
/// # Arguments
/// * `a` - First input matrix of integers
/// * `b` - Second input matrix of integers
///
/// # Returns
/// 2D convolution result as a matrix of integers
pub fn convolve_int_2d(a: &Vec<&Vec<u64>>, b: &Vec<&Vec<u64>>) -> Vec<Vec<u64>> {
    let a_as_complex = a
        .iter()
        .map(|row| row.iter().map(|&x| Complex::new(x as f64, 0.0)).collect())
        .collect();
    let b_as_complex = b
        .iter()
        .map(|row| row.iter().map(|&x| Complex::new(x as f64, 0.0)).collect())
        .collect();
    let convolved = convolve_complex_2d(a_as_complex, b_as_complex);
    convolved
        .into_iter()
        .map(|row| row.into_iter().map(|c| c.re.round() as u64).collect())
        .collect()
}
