use num_complex::Complex;

use std::ops::{Add, Mul, Neg, Sub};

use crate::key::{KeyPackage, PrivateKey};
use crate::ring::{N, Poly};

pub struct Ciphertext<'a> {
    scale_exp: usize,
    level: usize,
    c1: Poly,
    c2: Poly,
    keys: &'a KeyPackage,
    size: f64,
    noise: f64,
}

impl<'a> Ciphertext<'a> {
    pub fn from_message(
        message: [Complex<f64>; N / 2],
        scale_exp: usize,
        keys: &'a KeyPackage,
        noise: f64,
    ) -> Self {
        let bits = keys.bits();
        let scale_as_float = (1 << scale_exp) as f64;
        let slots = std::array::from_fn(|i| message[i] * scale_as_float);
        let poly = Poly::from_slots(slots, bits);

        let pk1 = keys.public_key().k1();
        let pk2 = keys.public_key().k2();
        let u = Poly::random_small(bits, noise);
        let e1 = Poly::random_small(bits, noise);
        let e2 = Poly::random_small(bits, noise);
        let c1 = &(&poly + &(&u * pk1)) + &e1;
        let c2 = &(&u * pk2) + &e2;

        let size = message.iter().map(|c| c.norm()).fold(0.0, f64::max);

        Self {
            scale_exp,
            level: bits,
            c1,
            c2,
            keys,
            size,
            noise,
        }
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn c1(&self) -> &Poly {
        &self.c1
    }

    pub fn c2(&self) -> &Poly {
        &self.c2
    }

    pub fn keys(&self) -> &'a KeyPackage {
        self.keys
    }

    pub fn size(&self) -> f64 {
        self.size
    }

    pub fn noise(&self) -> f64 {
        self.noise
    }

    pub fn decrypt(&self, private_key: &PrivateKey) -> [Complex<f64>; N / 2] {
        let level = self.level();
        let mut s = private_key.secret().clone();
        s.change_modulus(level);
        let c1 = &self.c1;
        let c2 = &self.c2;
        let poly = c1 + &(c2 * &s);
        let slots = poly.slots();
        let scale_as_float = (1 << self.scale_exp) as f64;
        std::array::from_fn(|i| slots[i] / scale_as_float)
    }

    fn rescale(&self, bits: usize) {
        unimplemented!()
    }

    fn coefficients_to_slots(&self) -> Self {
        unimplemented!()
    }

    fn slots_to_coefficients(&self) -> Self {
        unimplemented!()
    }

    fn refresh_level(&self) {
        unimplemented!()
    }

    fn bit_clean(&self) {
        unimplemented!()
    }

    fn integer_clean(&self) {
        unimplemented!()
    }
}

impl<'a> Add for &Ciphertext<'a> {
    type Output = Ciphertext<'a>;

    fn add(self, other: &Ciphertext<'a>) -> Ciphertext<'a> {
        assert!(std::ptr::eq(self.keys(), other.keys()));
        assert!(self.level == other.level);
        assert!(self.scale_exp == other.scale_exp);
        let keys = self.keys();
        let c1 = &self.c1 + &other.c1;
        let c2 = &self.c2 + &other.c2;
        let size = self.size() + other.size();
        let noise = self.noise() + other.noise();
        Ciphertext {
            scale_exp: self.scale_exp,
            level: self.level,
            c1,
            c2,
            keys,
            size,
            noise,
        }
    }
}

impl<'a> Sub for &Ciphertext<'a> {
    type Output = Ciphertext<'a>;

    fn sub(self, other: &Ciphertext<'a>) -> Ciphertext<'a> {
        assert!(std::ptr::eq(self.keys(), other.keys()));
        self + &(-other)
    }
}

impl<'a> Mul for &Ciphertext<'a> {
    type Output = Ciphertext<'a>;

    fn mul(self, other: &Ciphertext<'a>) -> Ciphertext<'a> {
        assert!(std::ptr::eq(self.keys(), other.keys()));
        assert!(self.level == other.level);
        assert!(self.scale_exp == other.scale_exp);
        let keys = self.keys();
        let evaluation_key = keys.evaluation_key();
        let ek1 = evaluation_key.k1();
        let ek2 = evaluation_key.k2();
        let p_exp = keys.p_exp();
        let level = self.level;
        let scale_exp = self.scale_exp;

        let c1p = self.c1() * other.c1();
        let c2p = &(self.c1() * other.c2()) + &(self.c2() * other.c1());
        let mut c3p = self.c2() * other.c2();
        c3p.change_modulus(level + p_exp);
        let mut c3p_times_ek1_over_p = &c3p * ek1;
        let mut c3p_times_ek2_over_p = &c3p * ek2;
        c3p_times_ek1_over_p.decrease_precision(p_exp);
        c3p_times_ek2_over_p.decrease_precision(p_exp);

        let mut c1 = &c1p + &c3p_times_ek1_over_p;
        let mut c2 = &c2p + &c3p_times_ek2_over_p;
        c1.decrease_precision(scale_exp);
        c2.decrease_precision(scale_exp);

        Ciphertext {
            scale_exp,
            level: level - scale_exp,
            c1,
            c2,
            keys: self.keys(),
            size: self.size(),
            noise: self.noise(),
        }
    }
}

impl<'a> Neg for &Ciphertext<'a> {
    type Output = Ciphertext<'a>;

    fn neg(self) -> Ciphertext<'a> {
        let keys = self.keys();
        let c1 = -&self.c1;
        let c2 = -&self.c2;
        let size = self.size();
        let noise = self.noise();
        Ciphertext {
            scale_exp: self.scale_exp,
            level: self.level,
            c1,
            c2,
            keys,
            size,
            noise,
        }
    }
}
