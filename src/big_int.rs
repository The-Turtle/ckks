use rand::Rng;
use std::fmt;
use std::ops::{Add, Mul, Shl, Shr};

use crate::fft::convolve_int;

pub const BITS_PER_LIMB: usize = 17;

#[derive(Debug, Clone)]
pub struct BigInt {
    limbs: Vec<u64>,
}

impl BigInt {
    pub fn from_limbs(limbs: Vec<u64>) -> Self {
        let mut big_int = Self::from_limbs_unchecked(limbs);
        big_int.carry();
        big_int
    }

    fn from_limbs_unchecked(limbs: Vec<u64>) -> Self {
        debug_assert!(limbs.len() > 0, "limbs must not be empty");
        Self { limbs }
    }

    pub fn from_int(value: u64) -> Self {
        let mut limbs = vec![0; 1];
        limbs[0] = value;
        Self::from_limbs(limbs)
    }

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

    pub fn zero() -> Self {
        Self::from_int(0)
    }

    pub fn limb_count(&self) -> usize {
        self.limbs.len()
    }

    pub fn get_limb(&self, index: usize) -> u64 {
        self.limbs.get(index).copied().unwrap_or(0)
    }

    pub fn get_limbs(&self) -> &Vec<u64> {
        &self.limbs
    }

    pub fn fits_in_bits(&self, bits: usize) -> bool {
        return (self >> bits).is_zero();
    }

    pub fn as_int(&self) -> u64 {
        debug_assert!(self.fits_in_bits(64), "BigInt does not fit in 64 bits");
        let mut result = 0;
        for i in 0..self.limb_count() {
            if i * BITS_PER_LIMB < 64 {
                result |= self.get_limb(i) << (i * BITS_PER_LIMB);
            }
        }
        result
    }

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

    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&limb| limb == 0)
    }

    pub fn increment(&mut self) {
        self.limbs[0] += 1;
        self.carry();
    }

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

    pub fn flipped(&self, bits: usize) -> BigInt {
        let mut flipped = self.clone();
        flipped.flip(bits);
        flipped
    }

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

    fn mul(self, other: &BigInt) -> BigInt {
        let a = &self.limbs;
        let b = &other.limbs;
        let c = convolve_int(a, b);
        BigInt::from_limbs(c)
    }
}

impl PartialEq for BigInt {
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut string = String::new();
        string.push_str("0x");
        string.push_str(&self.to_hex());
        write!(f, "{}", string)
    }
}
