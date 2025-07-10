use num_complex::Complex;
use std::fmt;
use std::ops::{Add, Mul, Neg, Shl};

use crate::big_int::BigInt;
use crate::big_mod::BigMod;
use crate::fft::{convolve_int_2d, fft, ifft, root_of_unity};
use crate::util::mod_power_of_2_inverse;

const N_EXP: usize = 10;
pub const N: usize = 1 << N_EXP;

#[derive(Debug, Clone)]
pub struct Poly {
    bits: usize,
    coeffs: [BigMod; N],
}

impl Poly {
    pub fn from_coeffs(coeffs: [BigMod; N]) -> Self {
        let bits = coeffs[0].bits();
        debug_assert!(
            coeffs.iter().all(|c| c.bits() == bits),
            "Not all coeffs have the same number of bits"
        );
        Self::from_coeffs_unchecked(coeffs)
    }

    fn from_coeffs_unchecked(coeffs: [BigMod; N]) -> Self {
        Self {
            bits: coeffs[0].bits(),
            coeffs,
        }
    }

    pub fn from_slots(slots: [Complex<f64>; N / 2], bits: usize) -> Self {
        let mut reversed_conj_slots = [Complex::<f64>::new(0.0, 0.0); N / 2];
        for (i, slot) in slots.iter().enumerate() {
            reversed_conj_slots[N / 2 - 1 - i] = slot.conj();
        }
        let mut interleaved = Vec::with_capacity(N);
        for i in 0..N / 2 {
            interleaved.push(slots[i]);
            interleaved.push(reversed_conj_slots[i]);
        }
        let transform_result = ifft(interleaved);
        let mut coeffs_as_ints = Vec::<i64>::with_capacity(N);
        for i in 0..N {
            let zeta = root_of_unity(-(i as i64), N_EXP + 1);
            let value = (zeta * transform_result[i] / N as f64).re.round();
            coeffs_as_ints.push(value as i64);
        }
        let coeffs_as_bigmod: [BigMod; N] =
            std::array::from_fn(|i| BigMod::from_int(bits, coeffs_as_ints[i]));
        Self::from_coeffs_unchecked(coeffs_as_bigmod)
    }

    pub fn from_int(bits: usize, value: i64) -> Self {
        let coeffs = std::array::from_fn(|i| {
            if i == 0 {
                BigMod::from_int(bits, value)
            } else {
                BigMod::zero(bits)
            }
        });
        Self::from_coeffs_unchecked(coeffs)
    }

    pub fn random(bits: usize) -> Self {
        let coeffs: [BigMod; N] = std::array::from_fn(|_| BigMod::random(bits));
        Self::from_coeffs_unchecked(coeffs)
    }

    pub fn random_small(bits: usize, std_dev: f64) -> Self {
        let coeffs: [BigMod; N] = std::array::from_fn(|_| BigMod::random_small(bits, std_dev));
        Self::from_coeffs_unchecked(coeffs)
    }

    pub fn bits(&self) -> usize {
        self.bits
    }

    pub fn coeff(&self, index: usize) -> &BigMod {
        debug_assert!(index < N, "Index out of bounds");
        &self.coeffs[index]
    }

    pub fn slots(&self) -> [Complex<f64>; N / 2] {
        let coeffs_as_ints: [i64; N] = std::array::from_fn(|i| self.coeffs[i].as_int());
        let mut transform_input = Vec::<Complex<f64>>::with_capacity(N);
        for i in 0..N {
            transform_input.push(coeffs_as_ints[i] as f64 * root_of_unity(i as i64, N_EXP + 1));
        }
        let transform_output = fft(transform_input);
        let mut slots = [Complex::new(0.0, 0.0); N / 2];
        for i in 0..N / 2 {
            slots[i] = transform_output[2 * i];
        }
        slots
    }

    pub fn decrease_precision(&mut self, bits: usize) {
        debug_assert!(bits <= self.bits);
        for coeff in &mut self.coeffs {
            coeff.decrease_precision(bits);
        }
        self.bits -= bits;
    }

    pub fn change_modulus(&mut self, new_modulus_exponent: usize) {
        for coeff in &mut self.coeffs {
            coeff.change_exponent(new_modulus_exponent);
        }
        self.bits = new_modulus_exponent;
    }

    pub fn substitution(&self, exponent: i64) -> Self {
        debug_assert!(exponent & 1 == 1);
        let mask = 2 * N - 1;
        let k = exponent as usize & mask;
        let k_inv = mod_power_of_2_inverse(k, N_EXP + 1).unwrap();
        let new_coeffs: [BigMod; N] = std::array::from_fn(|i| {
            let j = (k_inv * i) % (2 * N);
            if j < N {
                self.coeff(j).clone()
            } else {
                -self.coeff(j - N)
            }
        });
        Self::from_coeffs_unchecked(new_coeffs)
    }
}

impl Add for &Poly {
    type Output = Poly;

    fn add(self, other: &Poly) -> Poly {
        debug_assert!(self.bits() == other.bits());
        let coeffs: [BigMod; N] = std::array::from_fn(|i| self.coeff(i) + other.coeff(i));
        Poly::from_coeffs_unchecked(coeffs)
    }
}

impl Mul for &Poly {
    type Output = Poly;

    fn mul(self, other: &Poly) -> Poly {
        debug_assert!(self.bits() == other.bits());
        let bits = self.bits();

        let mut self_limb_table = Vec::with_capacity(N);
        let mut other_limb_table = Vec::with_capacity(N);
        for coeff in &self.coeffs {
            self_limb_table.push(coeff.residue().get_limbs());
        }
        for coeff in &other.coeffs {
            other_limb_table.push(coeff.residue().get_limbs());
        }

        let convolution = convolve_int_2d(&self_limb_table, &other_limb_table);
        let mut unreduced_coeffs: Vec<BigMod> = convolution
            .into_iter()
            .map(|limbs| BigMod::from_big_int(bits, BigInt::from_limbs(limbs)))
            .collect();
        unreduced_coeffs.resize(2 * N, BigMod::zero(bits));
        let coeffs: [BigMod; N] =
            std::array::from_fn(|i| &unreduced_coeffs[i] - &unreduced_coeffs[i + N]);
        Poly::from_coeffs_unchecked(coeffs)
    }
}

impl Neg for &Poly {
    type Output = Poly;

    fn neg(self) -> Poly {
        let coeffs = std::array::from_fn(|i| -&self.coeffs[i]);
        Poly::from_coeffs_unchecked(coeffs)
    }
}

impl Shl<usize> for &Poly {
    type Output = Poly;

    fn shl(self, other: usize) -> Poly {
        let shifted_coeffs = std::array::from_fn(|i| &self.coeffs[i] << other);
        Poly::from_coeffs_unchecked(shifted_coeffs)
    }
}

impl fmt::Display for Poly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let variable = "X";
        for (i, coeff) in self.coeffs.iter().enumerate().rev() {
            if i != self.coeffs.len() - 1 {
                write!(f, " + ")?;
            }
            if i == 0 {
                write!(f, "{}", coeff)?;
            } else if i == 1 {
                write!(f, "{}{}", coeff, variable)?;
            } else {
                write!(f, "{}{}^{}", coeff, variable, i)?;
            }
        }
        let bits = self.bits();
        write!(f, " (mod 2^{})", bits)
    }
}
