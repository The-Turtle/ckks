use std::fmt;
use std::ops::{Add, Mul, Neg, Shl, Shr, Sub};

use crate::big_int::BigInt;
use crate::util::gaussian;

#[derive(Debug, Clone)]
pub struct BigMod {
    bits: usize,
    residue: BigInt,
}

impl BigMod {
    pub fn from_big_int(bits: usize, mut residue: BigInt) -> Self {
        residue.truncate(bits);
        Self::from_big_int_unchecked(bits, residue)
    }

    fn from_big_int_unchecked(bits: usize, residue: BigInt) -> Self {
        BigMod { bits, residue }
    }

    pub fn from_int(bits: usize, value: i64) -> Self {
        let mut residue = BigInt::from_int(value.abs() as u64);
        if value < 0 {
            residue.flip(bits);
            residue.increment();
        }
        Self::from_big_int_unchecked(bits, residue)
    }

    pub fn random(bits: usize) -> Self {
        let residue = BigInt::random(bits);
        Self::from_big_int_unchecked(bits, residue)
    }

    pub fn random_small(bits: usize, std_dev: f64) -> Self {
        let x = gaussian(0.0, std_dev);
        let n = x.round() as i64;
        Self::from_int(bits, n)
    }

    pub fn zero(bits: usize) -> Self {
        Self::from_int(bits, 0)
    }

    pub fn bits(&self) -> usize {
        self.bits
    }

    pub fn residue(&self) -> &BigInt {
        &self.residue
    }

    pub fn as_int(&self) -> i64 {
        let negative = -self;
        if negative.residue < self.residue {
            if negative.residue.fits_in_bits(63) {
                return -(negative.residue.as_int() as i64);
            }
        } else {
            if self.residue.fits_in_bits(63) {
                return self.residue.as_int() as i64;
            }
        }
        panic!("Could not convert {} to i64", self);
    }

    pub fn decrease_precision(&mut self, bits: usize) {
        debug_assert!(bits <= self.bits);
        self.residue = &self.residue >> bits;
        self.bits -= bits;
    }

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

    fn add(self, other: &BigMod) -> BigMod {
        let bits = self.bits();
        let sum = self.residue() + other.residue();
        debug_assert!(self.bits() == other.bits());
        BigMod::from_big_int(bits, sum)
    }
}

impl Mul for &BigMod {
    type Output = BigMod;

    fn mul(self, other: &BigMod) -> BigMod {
        debug_assert!(self.bits() == other.bits());
        let bits = self.bits();
        let product = self.residue() * other.residue();
        BigMod::from_big_int(bits, product)
    }
}

impl Sub for &BigMod {
    type Output = BigMod;

    fn sub(self, other: &BigMod) -> BigMod {
        debug_assert!(self.bits() == other.bits());
        self + &-other
    }
}

impl Neg for &BigMod {
    type Output = BigMod;

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

    fn shl(self, other: usize) -> BigMod {
        let shifted_residue = self.residue() << other;
        let shifted_bits = self.bits() + other;
        BigMod::from_big_int(shifted_bits, shifted_residue)
    }
}

impl fmt::Display for BigMod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.residue)
    }
}
