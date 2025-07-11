//! Big integer implementation for arbitrary size arithmetic.
//!
//! This module provides a `BigInt` type that can handle nonnegative integers of
//! arbitrary size using a limb-based representation. Each limb stores a fixed
//! number of bits, and the implementation includes basic arithmetic operations,
//! bitwise operations, and utility functions for working with large integers.

use rand::Rng;
use std::fmt;
use std::ops::{Add, Mul, Shl, Shr};

use crate::error::ArithmeticError;
use crate::fft::convolve_int;

/// Number of bits stored in each limb of the BigInt.
///
/// This value determines the base of the internal representation. Each limb
/// can store values from 0 to 2^BITS_PER_LIMB - 1. The value recommended
/// by the CKKS paper is 19, which they claim is the largest possible value
/// for which ring multiplication using FFT does not give rounding errors.
/// In practice, this is currently set to 17 because I have not figured out
/// how to decrease the error in the FFT algorithm.
pub const BITS_PER_LIMB: usize = 17;

/// A big integer type that can represent arbitrarily large integers.
///
/// `BigInt` uses a limb-based representation where each limb stores
/// `BITS_PER_LIMB` bits. The limbs are stored in little-endian order,
/// meaning the least significant limb is at index 0.
#[derive(Debug, Clone)]
pub struct BigInt {
    /// The limbs of the big integer, stored in little-endian order.
    /// Each limb contains at most `BITS_PER_LIMB` bits.
    limbs: Vec<u64>,
}

impl BigInt {
    /// Creates a new `BigInt` from a vector of limbs.
    ///
    /// This function performs carry propagation to ensure each limb
    /// contains at most `BITS_PER_LIMB` bits, so it is not necessary
    /// that the input limbs are each at most `BITS_PER_LIMB` bits.
    ///
    /// # Arguments
    ///
    /// * `limbs` - A vector of limbs in little-endian order
    ///
    /// # Panics
    ///
    /// Panics if the limbs vector is empty (in debug mode).
    pub fn from_limbs(limbs: Vec<u64>) -> Self {
        let mut big_int = Self::from_limbs_unchecked(limbs);
        big_int.carry();
        big_int
    }

    /// Creates a new `BigInt` from limbs without carry propagation.
    ///
    /// This is an internal function that creates a `BigInt` without
    /// performing carry operations. It should only be used when you
    /// know the limbs are already in the correct format.
    ///
    /// # Arguments
    ///
    /// * `limbs` - A vector of limbs in little-endian order
    ///
    /// # Panics
    ///
    /// Panics if the limbs vector is empty (in debug mode).
    fn from_limbs_unchecked(limbs: Vec<u64>) -> Self {
        debug_assert!(limbs.len() > 0, "limbs must not be empty");
        Self { limbs }
    }

    /// Creates a new `BigInt` from a 64-bit unsigned integer.
    ///
    /// # Arguments
    ///
    /// * `value` - The integer value to convert
    pub fn from_int(value: u64) -> Self {
        let mut limbs = vec![0; 1];
        limbs[0] = value;
        Self::from_limbs(limbs)
    }

    /// Creates a random `BigInt` with the specified number of bits.
    ///
    /// The generated number will have exactly the specified number of bits,
    /// with random values in each bit position. The distribution is uniform
    /// in the range [0, 2^bits).
    ///
    /// # Arguments
    ///
    /// * `bits` - The number of bits for the random integer
    pub fn random(bits: usize) -> Self {
        let mut rng = rand::thread_rng();
        let limb_count = 1 + (bits - 1) / BITS_PER_LIMB;
        let limbs: Vec<u64> = (0..limb_count)
            .map(|_| rng.r#gen::<u64>() & ((1 << BITS_PER_LIMB) - 1))
            .collect();
        let mut big_int = Self::from_limbs_unchecked(limbs);
        big_int.truncate(bits);
        big_int
    }

    /// Creates a `BigInt` representing zero. This is usually used for
    /// initializing arrays of BigInts.
    pub fn zero() -> Self {
        Self::from_int(0)
    }

    /// An internal function to get the number of limbs in this `BigInt`.
    fn limb_count(&self) -> usize {
        self.limbs.len()
    }

    /// A private function that returns the value of the limb at the given index.
    ///
    /// If the index is out of bounds, returns 0. This allows for
    /// convenient access to limbs without bounds checking.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the limb to retrieve
    fn get_limb(&self, index: usize) -> u64 {
        self.limbs.get(index).copied().unwrap_or(0)
    }

    /// Returns a reference to the internal limbs vector.
    ///
    /// This function should only be used for functions that need
    /// direct access to the underlying representation for
    /// performance-critical operations, such as ring multiplication
    /// via FFT.
    pub fn get_limbs(&self) -> &Vec<u64> {
        &self.limbs
    }

    /// Checks if this `BigInt` fits within the specified number of bits.
    ///
    /// Returns `true` if the number is in the range [0, 2^bits - 1] and
    /// `false` otherwise.
    ///
    /// # Arguments
    ///
    /// * `bits` - The number of bits to check against
    pub fn fits_in_bits(&self, bits: usize) -> bool {
        return (self >> bits).is_zero();
    }

    /// Converts this `BigInt` to a 64-bit integer if it is
    /// at most 64 bits long.
    ///
    /// Returns `Err(ArithmeticError::Overflow)` if the `BigInt`
    /// doesn't fit in 64 bits.
    pub fn as_int(&self) -> Result<u64, ArithmeticError> {
        if !self.fits_in_bits(64) {
            return Err(ArithmeticError::Overflow);
        }

        let mut result = 0;
        for i in 0..self.limb_count() {
            if i * BITS_PER_LIMB < 64 {
                result |= self.get_limb(i) << (i * BITS_PER_LIMB);
            }
        }
        Ok(result)
    }

    /// Performs carry propagation on the limbs. This ensures that each
    /// limb contains at most `BITS_PER_LIMB` bits and removes leading
    /// zero limbs (except for the last one, which only occurs when the
    /// `BigInt` is zero).
    ///
    /// This is an internal function and should only be called by other
    /// methods in this module. No outside code should need to call this
    /// function because all public methods should handle carry
    /// propagation internally.
    fn carry(&mut self) {
        let mut carry = 0;
        let mask = (1 << BITS_PER_LIMB) - 1;

        for limb in &mut self.limbs {
            *limb += carry;
            carry = *limb >> BITS_PER_LIMB;
            *limb &= mask;
        }

        loop {
            if carry == 0 {
                break;
            }
            self.limbs.push(carry & mask);
            carry = carry >> BITS_PER_LIMB;
        }

        while self.limbs.len() > 1 && *self.limbs.last().unwrap() == 0 {
            self.limbs.pop();
        }
    }

    /// Checks if this `BigInt` is zero.
    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&limb| limb == 0)
    }

    /// Increments this `BigInt` by one.
    pub fn increment(&mut self) {
        self.limbs[0] += 1;
        self.carry();
    }

    /// Truncates this `BigInt` to the specified number of bits, so
    /// the result lies in the range [0, 2^bits). This is equivalent
    /// to ANDing with a mask that has the specified number of bits
    /// set to 1.
    ///
    /// All bits beyond the specified count are set to zero.
    pub fn truncate(&mut self, bits: usize) {
        let limb_count = 1 + (bits - 1) / BITS_PER_LIMB;
        if limb_count <= self.limb_count() {
            self.limbs.truncate(limb_count);
            if bits % BITS_PER_LIMB != 0 {
                if let Some(last) = self.limbs.last_mut() {
                    let mask = (1 << bits % BITS_PER_LIMB) - 1;
                    *last &= mask;
                }
            }
        }
    }

    /// Flips the specified number of least significant bits. This
    /// is equivalent to XORing with a mask that has the specified
    /// number of bits set to 1.
    ///
    /// # Arguments
    ///
    /// * `bits` - The number of bits to flip
    pub fn flip(&mut self, bits: usize) {
        let limb_count = 1 + (bits - 1) / BITS_PER_LIMB;
        if limb_count > self.limb_count() {
            self.limbs.resize(limb_count, 0);
        }
        let mask = (1 << BITS_PER_LIMB) - 1;
        let limb_flips = bits / BITS_PER_LIMB;
        let bit_flips = bits % BITS_PER_LIMB;

        for limb in self.limbs.iter_mut().take(limb_flips) {
            *limb ^= mask;
        }

        if bit_flips > 0 {
            let final_mask = (1 << bit_flips) - 1;
            self.limbs[limb_flips] ^= final_mask;
        }
    }

    /// Returns a new `BigInt` with the specified number of bits flipped.
    ///
    /// This is a non-mutating version of `flip()`.
    ///
    /// # Arguments
    ///
    /// * `bits` - The number of bits to flip
    pub fn flipped(&self, bits: usize) -> BigInt {
        let mut flipped = self.clone();
        flipped.flip(bits);
        flipped
    }

    /// Converts this `BigInt` to a hexadecimal string.
    ///
    /// The returned string contains only the hexadecimal digits without
    /// the "0x" prefix.
    pub fn to_hex(&self) -> String {
        let mut bits = Vec::new();
        for &limb in self.limbs.iter().rev() {
            for i in (0..BITS_PER_LIMB).rev() {
                bits.push(((limb >> i) & 1) != 0);
            }
        }
        while bits.first() == Some(&false) {
            bits.remove(0);
        }
        while bits.len() % 4 != 0 {
            bits.insert(0, false);
        }
        let mut string = String::new();
        if bits.is_empty() {
            string.push_str("0");
        }
        for chunk in bits.chunks(4) {
            let mut value = 0u8;
            for &bit in chunk {
                value = (value << 1) | (bit as u8);
            }
            string.push_str(&format!("{:x}", value));
        }
        string
    }
}

impl Add for &BigInt {
    type Output = BigInt;

    /// Adds two `BigInt` values.
    fn add(self, other: &BigInt) -> BigInt {
        let max_len = std::cmp::max(self.limb_count(), other.limb_count());
        let mut summed_limbs = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = self.get_limb(i);
            let b = other.get_limb(i);
            summed_limbs.push(a + b);
        }
        BigInt::from_limbs(summed_limbs)
    }
}

impl Mul for &BigInt {
    type Output = BigInt;

    /// Multiplies two `BigInt` values.
    /// This uses FFT-based convolution, so if there are too many limbs,
    /// the multiplication might be incorrect due to imprecision of the FFT algorithm.
    /// To fix this issue, decrease BITS_PER_LIMB in big_int.rs or decrease the number
    /// of limbs in the BigInts.
    fn mul(self, other: &BigInt) -> BigInt {
        let a = &self.limbs;
        let b = &other.limbs;
        let c = convolve_int(a, b);
        BigInt::from_limbs(c)
    }
}

impl PartialEq for BigInt {
    /// Compares two `BigInt` values for equality.
    fn eq(&self, other: &Self) -> bool {
        let max_len = std::cmp::max(self.limb_count(), other.limb_count());
        for i in 0..max_len {
            if self.get_limb(i) != other.get_limb(i) {
                return false;
            }
        }
        true
    }
}

impl Eq for BigInt {}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigInt {
    /// Compares two `BigInt` values for ordering.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_len = self.limb_count();
        let other_len = other.limb_count();
        let max_len = std::cmp::max(self_len, other_len);

        for i in (0..max_len).rev() {
            let a = self.get_limb(i);
            let b = other.get_limb(i);
            if a < b {
                return std::cmp::Ordering::Less;
            } else if a > b {
                return std::cmp::Ordering::Greater;
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl Shl<usize> for &BigInt {
    type Output = BigInt;

    /// Left-shifts a `BigInt` by the specified number of bits.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The number of bits to shift left
    fn shl(self, rhs: usize) -> BigInt {
        let bit_shift = rhs % BITS_PER_LIMB;
        let limb_shift = rhs / BITS_PER_LIMB;
        let mut shifted_limbs = vec![0; self.limb_count() + limb_shift + 1];

        let mask = (1 << BITS_PER_LIMB) - 1;
        for i in 0..(self.limb_count()) {
            let limb = self.get_limb(i);
            let a = (limb << bit_shift) & mask;
            let b = limb >> (BITS_PER_LIMB - bit_shift);
            shifted_limbs[i + limb_shift] |= a;
            shifted_limbs[i + limb_shift + 1] |= b;
        }

        BigInt::from_limbs_unchecked(shifted_limbs)
    }
}

impl Shr<usize> for &BigInt {
    type Output = BigInt;

    /// Right-shifts a `BigInt` by the specified number of bits.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The number of bits to shift right
    fn shr(self, rhs: usize) -> BigInt {
        let bit_shift = rhs % BITS_PER_LIMB;
        let limb_shift = rhs / BITS_PER_LIMB;

        if limb_shift >= self.limb_count() {
            return BigInt::zero();
        }

        let mut shifted_limbs = Vec::with_capacity(self.limb_count() - limb_shift);

        for i in limb_shift..self.limb_count() {
            let a = self.get_limb(i) >> bit_shift;
            let b = self.get_limb(i + 1) & ((1 << bit_shift) - 1);
            shifted_limbs.push(a | b << (BITS_PER_LIMB - bit_shift));
        }
        BigInt::from_limbs_unchecked(shifted_limbs)
    }
}

impl fmt::Display for BigInt {
    /// Formats the `BigInt` as a hexadecimal string with "0x" prefix.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut string = String::new();
        string.push_str("0x");
        string.push_str(&self.to_hex());
        write!(f, "{}", string)
    }
}
