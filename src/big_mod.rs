use std::fmt;
use std::ops::{Add, Mul, Neg, Shl, Sub};

use crate::big_int::BigInt;
use crate::error::ArithmeticError;
use crate::util::gaussian;

/// A big integer modulo 2^bits.
///
/// This struct represents an integer in the ring Z/(2^bits)Z.
#[derive(Debug, Clone)]
pub struct BigMod {
    /// The number of bits in the modulus (so the modulus is 2^bits)
    bits: usize,
    /// The residue value, always kept in the range [0, 2^bits)
    residue: BigInt,
}

impl BigMod {
    /// Creates a BigMod from a BigInt, truncating to fit within the specified bit size.
    ///
    /// # Arguments
    /// * `bits` - The number of bits for the modulus (modulo 2^bits)
    /// * `residue` - The BigInt value to be reduced modulo 2^bits
    pub fn from_big_int(bits: usize, mut residue: BigInt) -> Self {
        residue.truncate(bits);
        Self::from_big_int_unchecked(bits, residue)
    }

    /// Creates a BigMod from a BigInt without checking if it fits in the bit size.
    /// This should only be called by internal functions that ensure the residue is
    /// properly reduced modulo 2^bits.
    ///
    /// # Safety
    /// The caller must ensure that the residue is already properly reduced modulo 2^bits.
    fn from_big_int_unchecked(bits: usize, residue: BigInt) -> Self {
        BigMod { bits, residue }
    }

    /// Creates a BigMod from a signed integer.
    ///
    /// # Arguments
    /// * `bits` - The number of bits for the modulus
    /// * `value` - The signed integer value to convert
    pub fn from_int(bits: usize, value: i64) -> Self {
        let mut residue = BigInt::from_int(value.abs() as u64);
        if value < 0 {
            residue.flip(bits);
            residue.increment();
        }
        Self::from_big_int_unchecked(bits, residue)
    }

    /// Creates a random BigMod with uniform distribution over [0, 2^bits).
    ///
    /// # Arguments
    /// * `bits` - The number of bits for the modulus
    pub fn random(bits: usize) -> Self {
        let residue = BigInt::random(bits);
        Self::from_big_int_unchecked(bits, residue)
    }

    /// Creates a random BigMod with Gaussian distribution around zero,
    /// rounded to the nearest integer.
    ///
    /// # Arguments
    /// * `bits` - The number of bits for the modulus
    /// * `std_dev` - The standard deviation of the Gaussian distribution
    pub fn random_small(bits: usize, std_dev: f64) -> Self {
        let x = gaussian(0.0, std_dev);
        let n = x.round() as i64;
        Self::from_int(bits, n)
    }

    /// Creates a BigMod representing zero.
    ///
    /// # Arguments
    /// * `bits` - The number of bits for the modulus
    pub fn zero(bits: usize) -> Self {
        Self::from_int(bits, 0)
    }

    /// Returns the number of bits in the modulus.
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// Returns a reference to the underlying BigInt residue.
    pub fn residue(&self) -> &BigInt {
        &self.residue
    }

    /// Converts the BigMod to a signed integer.
    ///
    /// This method determines whether the value should be interpreted as positive or negative
    /// by comparing it with its negation. If the negation has a smaller residue, the original
    /// value is interpreted as negative.
    ///
    /// # Returns
    /// * `Ok(i64)` - The signed integer representation
    /// * `Err(ArithmeticError)` - If the value is too large to fit in an i64
    pub fn as_int(&self) -> Result<i64, ArithmeticError> {
        let negative = -self;
        if negative.residue < self.residue {
            let result = negative.residue.as_int();
            match result {
                Ok(negative_residue) => Ok(-(negative_residue as i64)),
                Err(other) => Err(other),
            }
        } else {
            let result = self.residue.as_int();
            match result {
                Ok(residue) => Ok(residue as i64),
                Err(other) => Err(other),
            }
        }
    }

    /// Decreases the precision by right-shifting the residue and reducing the bit count.
    /// This effectively divides the value by 2^bits and reduces the modulus size. In
    /// debug mode, this raises an error if right-shifting would make the bit count
    /// non-positive.
    ///
    /// # Arguments
    /// * `bits` - The number of bits to decrease the precision by
    pub fn decrease_precision(&mut self, bits: usize) {
        debug_assert!(bits < self.bits);
        self.residue = &self.residue >> bits;
        self.bits -= bits;
    }

    /// Changes the exponent (bit size) of the modulus. If the exponent is less
    /// than the bit count of the modulus, then the modulus is truncated. If the
    /// exponent is greater than the current bit count, then the modulus is chosen
    /// to preserve the signed value closest to zero. (If the modulus is 2^(bits-1)
    /// then it is interpreted as positive.)
    ///
    /// # Arguments
    /// * `new_exponent` - The new number of bits for the modulus
    pub fn change_exponent(&mut self, new_exponent: usize) {
        if new_exponent < self.bits {
            self.residue.truncate(new_exponent);
        }
        if self.residue.fits_in_bits(self.bits - 1) {
            self.bits = new_exponent;
        } else {
            let mut negative_self = -&*self;
            negative_self.bits = new_exponent;
            let negative_negative_self = -&negative_self;
            self.residue = negative_negative_self.residue;
            self.bits = new_exponent;
        }
    }
}

impl Add for &BigMod {
    type Output = BigMod;

    /// Addition operation for BigMod values.
    fn add(self, other: &BigMod) -> BigMod {
        let bits = self.bits();
        let sum = self.residue() + other.residue();
        debug_assert!(self.bits() == other.bits());
        BigMod::from_big_int(bits, sum)
    }
}

impl Mul for &BigMod {
    type Output = BigMod;

    /// Multiplication operation for BigMod values. If the BigInts
    /// have too many limbs, then the multiplication might be
    /// incorrect due to imprecision of the FFT algorithm. To
    /// fix this issue, decrease BITS_PER_LIMB in big_int.rs or
    /// decrease the number of limbs in the BigInts.
    fn mul(self, other: &BigMod) -> BigMod {
        debug_assert!(self.bits() == other.bits());
        let bits = self.bits();
        let product = self.residue() * other.residue();
        BigMod::from_big_int(bits, product)
    }
}

impl Sub for &BigMod {
    type Output = BigMod;

    /// Subtraction operation for BigMod values.
    fn sub(self, other: &BigMod) -> BigMod {
        debug_assert!(self.bits() == other.bits());
        self + &-other
    }
}

impl Neg for &BigMod {
    type Output = BigMod;

    /// Negation operation for BigMod values.
    fn neg(self) -> BigMod {
        if self.residue.is_zero() {
            return BigMod::zero(self.bits);
        }
        let mut negative = self.residue.flipped(self.bits);
        negative.increment();
        let bits = self.bits;
        BigMod::from_big_int_unchecked(bits, negative)
    }
}

impl Shl<usize> for &BigMod {
    type Output = BigMod;

    /// Left shift operation for BigMod values.
    ///
    /// Performs a left shift by the specified number of bits. The bit size of
    /// the result is increased to accommodate the shifted value. Applying
    /// decrease_precision reverses this operation.
    fn shl(self, other: usize) -> BigMod {
        let shifted_residue = self.residue() << other;
        let shifted_bits = self.bits() + other;
        BigMod::from_big_int(shifted_bits, shifted_residue)
    }
}

impl fmt::Display for BigMod {
    /// Formats the BigMod by displaying its underlying residue value. The
    /// modulus is not displayed.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.residue)
    }
}
