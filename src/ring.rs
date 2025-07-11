//! # Polynomial Ring Implementation for CKKS Encryption
//!
//! This module implements polynomial operations in the ring R = Z[X]/(X^N + 1) where N is a power of 2.
//! This is a core component of the CKKS (Cheon-Kim-Kim-Song) fully homomorphic encryption scheme.
//!
//! The polynomial ring supports:
//! - Addition and efficient multiplication of polynomials
//! - Conversion between coefficient representation and slot representation
//! - Precision management and modulus switching
//! - Substitution operations for automorphisms

use num_complex::Complex;
use std::fmt;
use std::ops::{Add, Mul, Neg, Shl};

use crate::big_int::BigInt;
use crate::big_mod::BigMod;
use crate::error::ArithmeticError;
use crate::fft::{convolve_int_2d, fft, ifft, root_of_unity};
use crate::util::mod_power_of_2_inverse;

/// Ring dimension exponent (N = 2^N_EXP).
/// This needs to be at least 10 for security.
const N_EXP: usize = 10;

/// Ring dimension - the degree of the polynomial ring
/// For CKKS, this is typically a power of 2 (here 1024).
/// In this implementation, the ring dimension is required
/// to be a power of 2.
pub const N: usize = 1 << N_EXP;

/// Polynomial in the ring Z[X]/(X^N + 1)
///
/// Represents a polynomial with coefficients in Z/qZ where q = 2^bits.
#[derive(Debug, Clone)]
pub struct Poly {
    /// Number of bits in the coefficient modulus
    bits: usize,
    /// Polynomial coefficients, stored in order from constant term to highest degree
    coeffs: [BigMod; N],
}

impl Poly {
    /// Create a polynomial from an array of coefficients
    ///
    /// # Arguments
    /// * `coeffs` - Array of N coefficients, given in ascending order
    /// from constant term to highest degree
    ///
    /// # Panics
    /// Panics in debug mode if the coefficients don't all have the same modulus
    pub fn from_coeffs(coeffs: [BigMod; N]) -> Self {
        let bits = coeffs[0].bits();
        debug_assert!(
            coeffs.iter().all(|c| c.bits() == bits),
            "Not all coeffs have the same number of bits"
        );
        Self::from_coeffs_unchecked(coeffs)
    }

    /// Create a polynomial from coefficients without checking that
    /// all coefficients have the same modulus. It is expected that the caller
    /// has ensured that all coefficients have the same modulus.
    fn from_coeffs_unchecked(coeffs: [BigMod; N]) -> Self {
        Self {
            bits: coeffs[0].bits(),
            coeffs,
        }
    }

    /// Creates a polynomial from slot values; i.e. evaluations on the
    /// primitive 2N-th roots of unity.
    ///
    /// If the slot inputs are z0, z1, ..., z(N/2-1), then the polynomial P
    /// approximately satisfies
    /// P(zeta) = z0
    /// P(zeta^5) = z1
    /// P(zeta^9) = z2
    /// ...
    /// P(zeta^(2N-3)) = z^(N/2-1)
    /// P(zeta^-1) = z0.conj()
    /// P(zeta^-5) = z1.conj()
    /// P(zeta^-9) = z2.conj()
    /// ...
    /// P(zeta^-(2N-3)) = z^(N/2-1).conj()
    /// where zeta = exp(2*pi*i/2*N) is a primitive 2N-th root of unity.
    /// Note that the coefficients of the interpolating polynomial will be rounded
    /// to the nearest integer, and the resulting coefficients will be taken modulo
    /// 2^bits. The method `get_slots()` approximately recovers the input
    /// to `from_slots`, assuming that the coefficients do not overflow the modulus.
    ///
    /// # Arguments
    /// * `slots` - Array of N/2 complex values representing the encoded data
    /// * `bits` - Number of bits for the coefficient modulus
    ///
    /// # Returns
    /// A polynomial whose slot representation matches the input
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

    /// Create a constant polynomial (polynomial with only a constant term)
    ///
    /// # Arguments
    /// * `bits` - Number of bits for the coefficient modulus
    /// * `value` - The constant coefficient
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

    /// Generate a random polynomial with coefficients uniformly distributed in Z/qZ
    ///
    /// # Arguments
    /// * `bits` - Number of bits for the coefficient modulus
    pub fn random(bits: usize) -> Self {
        let coeffs: [BigMod; N] = std::array::from_fn(|_| BigMod::random(bits));
        Self::from_coeffs_unchecked(coeffs)
    }

    /// Generate a random polynomial with small coefficients following a Gaussian distribution
    ///
    /// Used for generating secret keys and noise in CKKS encryption.
    ///
    /// # Arguments
    /// * `bits` - Number of bits for the coefficient modulus
    /// * `std_dev` - Standard deviation of the Gaussian distribution
    pub fn random_small(bits: usize, std_dev: f64) -> Self {
        let coeffs: [BigMod; N] = std::array::from_fn(|_| BigMod::random_small(bits, std_dev));
        Self::from_coeffs_unchecked(coeffs)
    }

    /// Get the number of bits in the coefficient modulus
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// Get a reference to a specific coefficient
    ///
    /// # Arguments
    /// * `index` - Index of the coefficient (0 = constant term, 1 = coefficient of X, etc.)
    ///
    /// # Panics
    /// Panics in debug mode if index is out of bounds
    pub fn coeff(&self, index: usize) -> &BigMod {
        debug_assert!(index < N, "Index out of bounds");
        &self.coeffs[index]
    }

    /// Evaluates the polynomial on the 2N^th roots of unity. Specifically,
    /// let P denote the polynomial with coefficients given by converting the
    /// BigMod coefficients to the range (2^(bits-1), 2^(bits-1)]. Then this
    /// function returns [P(zeta), P(zeta^5), P(zeta^9), ..., P(zeta^(2N-3))].
    /// This should approximately return the input from the constructor
    /// `from_slots`. If the BigMod coefficients cannot be converted to 64-bit
    /// integers, an error is returned.
    ///
    /// # Returns
    /// Array of N/2 complex values representing the polynomial in slot form,
    /// or an error if coefficients cannot be converted to integers
    pub fn slots(&self) -> Result<[Complex<f64>; N / 2], ArithmeticError> {
        let mut coeffs_as_ints: [i64; N] = [0i64; N];
        for i in 0..N {
            let result = self.coeffs[i].as_int();
            match result {
                Ok(coeff) => coeffs_as_ints[i] = coeff,
                Err(err) => return Err(err),
            }
        }

        let mut transform_input = Vec::<Complex<f64>>::with_capacity(N);
        for i in 0..N {
            transform_input.push(coeffs_as_ints[i] as f64 * root_of_unity(i as i64, N_EXP + 1));
        }

        let transform_output = fft(transform_input);
        let mut slots = [Complex::new(0.0, 0.0); N / 2];
        for i in 0..N / 2 {
            slots[i] = transform_output[2 * i];
        }
        Ok(slots)
    }

    /// Decrease the precision of all coefficients by the specified number of bits
    ///
    /// This is used for rescaling during CKKS multiplication.
    ///
    /// # Arguments
    /// * `bits` - Number of bits to remove from the precision
    ///
    /// # Panics
    /// Panics in debug mode if trying to remove more bits than available
    pub fn decrease_precision(&mut self, bits: usize) {
        debug_assert!(bits <= self.bits);
        for coeff in &mut self.coeffs {
            coeff.decrease_precision(bits);
        }
        self.bits -= bits;
    }

    /// Change the modulus of all coefficients to a new power of 2. If the
    /// new modulus is smaller than the current one, the coefficients are
    /// reduced modulo the new modulus. If the new modulus is larger than the
    /// current one, the coefficients are extended modulo the new modulus,
    /// preserving the representation that lies in the range (-2^(bits-1), 2^(bits-1)].
    ///
    /// # Arguments
    /// * `new_modulus_exponent` - The new modulus will be 2^new_modulus_exponent
    pub fn change_modulus(&mut self, new_modulus_exponent: usize) {
        for coeff in &mut self.coeffs {
            coeff.change_exponent(new_modulus_exponent);
        }
        self.bits = new_modulus_exponent;
    }

    /// Apply a substitution X -> X^exponent to the polynomial
    ///
    /// This implements automorphisms of the polynomial ring, which are used
    /// for operations like rotations in CKKS.
    ///
    /// # Arguments
    /// * `exponent` - Must be odd; that is, coprime to 2^bits.
    ///
    /// # Panics
    /// Panics in debug mode if exponent is not odd
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

/// Addition of polynomials in the ring
///
/// Adds corresponding coefficients modulo the coefficient modulus.
/// Both polynomials must have the same bit precision.
impl Add for &Poly {
    type Output = Poly;

    fn add(self, other: &Poly) -> Poly {
        debug_assert!(self.bits() == other.bits());
        let coeffs: [BigMod; N] = std::array::from_fn(|i| self.coeff(i) + other.coeff(i));
        Poly::from_coeffs_unchecked(coeffs)
    }
}

/// Multiplication of polynomials in the ring Z[X]/(X^N + 1)
///
/// Performs polynomial multiplication with reduction modulo X^N + 1.
/// This uses FFT-based convolution for efficiency, so this could
/// return an incorrect value if N or the modulus is too large. In this case,
/// decrease the value of BITS_PER_LIMB in bit_int.rs.
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

/// Negation of polynomials
///
/// Negates all coefficients in the polynomial.
impl Neg for &Poly {
    type Output = Poly;

    fn neg(self) -> Poly {
        let coeffs = std::array::from_fn(|i| -&self.coeffs[i]);
        Poly::from_coeffs_unchecked(coeffs)
    }
}

/// Left shift operation (multiplication by 2^shift)
///
/// Multiplies each coefficient by 2^shift, effectively scaling the polynomial.
/// This also increases the modulus of the polynomial by a factor of 2^shift.
impl Shl<usize> for &Poly {
    type Output = Poly;

    fn shl(self, shift: usize) -> Poly {
        let shifted_coeffs = std::array::from_fn(|i| &self.coeffs[i] << shift);
        Poly::from_coeffs_unchecked(shifted_coeffs)
    }
}

/// Display formatting for polynomials
///
/// Shows the polynomial in standard mathematical notation with the modulus.
/// The coefficients are displayed in hexadecimal format.
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
