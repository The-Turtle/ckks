use num_complex::Complex;
use once_cell::sync::Lazy;
use std::f64::consts::PI;

const ROOTS_CACHE_SIZE: usize = 11;

const TAU: f64 = 2.0 * PI;

static ROOTS_OF_UNITY: Lazy<Vec<Complex<f64>>> = Lazy::new(|| {
    let n = 1 << ROOTS_CACHE_SIZE;
    (0..n)
        .map(|k| {
            let theta = TAU * (k as f64) / (n as f64);
            Complex::from_polar(1.0, theta)
        })
        .collect()
});

pub fn root_of_unity(k: i64, bits: usize) -> &'static Complex<f64> {
    debug_assert!(bits <= ROOTS_CACHE_SIZE);
    let mask = (1 << bits) - 1;
    let i = (k & mask) as usize;
    &ROOTS_OF_UNITY[i << (ROOTS_CACHE_SIZE - bits)]
}

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

pub fn fft(input: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    transform(input, false)
}

pub fn ifft(input: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    transform(input, true)
}

fn convolve_complex(mut a: Vec<Complex<f64>>, mut b: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let n = (a.len() + b.len() - 1).next_power_of_two();
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

fn fft_2d(input: Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    transform_2d(input, false)
}

fn ifft_2d(input: Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
    transform_2d(input, true)
}

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
