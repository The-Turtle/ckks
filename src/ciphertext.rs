use num_complex::Complex;

use std::ops::{Add, Mul, Neg, Sub};

use crate::error::CiphertextError;
use crate::key::{KeyPackage, PrivateKey};
use crate::ring::{N, Poly};

/// A ciphertext in the CKKS homomorphic encryption scheme.
///
/// This struct represents an encrypted message that can be operated on
/// homomorphically (addition, subtraction, multiplication) without decryption.
/// The ciphertext consists of two polynomials (c1, c2) and maintains metadata
/// about the encryption parameters. Each operation incurs some precision loss
/// due to the nature of homomorphic encryption; this precision can be refreshed
/// using bit bootstrapping or small integer bootstrapping, but only if the encrypted
/// data was discretized to begin with. Each operation also decreases the bit
/// count (modulus) by some fixed amount; the bit count should not decrease below
/// the minimum required for the coefficients of the polynomial encoding the message
/// to overflow the modulus. To reset the modulus, the ``refresh_bits'' method
/// can be used.
#[derive(Clone)]
pub struct Ciphertext<'a> {
    /// The scaling exponent used for fixed-point representation. If the message
    /// is [m0, m1, ..., m_(N/2)], then the decrypted polynomial evaluated on
    /// the slots is [m0 * 2^scale_exp, m1 * 2^scale_exp, ..., m_(N/2) * 2^scale_exp].
    scale_exp: usize,
    /// The number of bits in the current modulus. This should start of at around
    /// 1000 and should never drop below 60.
    bits: usize,
    /// First polynomial component of the ciphertext
    c1: Poly,
    /// Second polynomial component of the ciphertext
    c2: Poly,
    /// Reference to the key package containing public and evaluation keys
    keys: &'a KeyPackage,
    /// The maximum absolute value of the encrypted message. This will
    /// increase during homomorphic operations.
    size: f64,
    /// The noise level in the ciphertext. This will increase during homomorphic
    /// operations, but can be reduced via bit bootstrapping or small integer
    /// bootstrapping.
    noise: f64,
}

impl<'a> Ciphertext<'a> {
    /// Creates a new ciphertext from a message.
    ///
    /// # Arguments
    /// * `message` - Array of complex numbers representing the plaintext message
    /// * `scale_exp` - Scaling exponent for fixed-point representation
    /// * `keys` - Reference to the key package containing encryption keys
    /// * `noise` - Noise parameter for encryption randomness
    ///
    /// # Returns
    /// A new `Ciphertext` containing the encrypted message
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
            bits: bits,
            c1,
            c2,
            keys,
            size,
            noise,
        }
    }

    /// Creates a new ciphertext from a complex number.
    ///
    /// # Arguments
    ///
    /// * `z` - The complex number to be encrypted.
    /// * `scale_exp` - The exponent of the scale factor.
    /// * `keys` - The key package used for encryption.
    /// * `noise` - The noise level for encryption.
    ///
    /// # Returns
    ///
    /// A new ciphertext that encodes N/2 copies of the complex number.
    pub fn from_complex(
        z: Complex<f64>,
        scale_exp: usize,
        keys: &'a KeyPackage,
        noise: f64,
    ) -> Self {
        let message: [Complex<f64>; N / 2] = [z; N / 2];
        Ciphertext::from_message(message, scale_exp, keys, noise)
    }

    /// Returns the number of bits in the current modulus.
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// Returns a reference to the first polynomial component of the ciphertext.
    pub fn c1(&self) -> &Poly {
        &self.c1
    }

    /// Returns a reference to the second polynomial component of the ciphertext.
    pub fn c2(&self) -> &Poly {
        &self.c2
    }

    /// Returns a reference to the key package used for this ciphertext.
    pub fn keys(&self) -> &'a KeyPackage {
        self.keys
    }

    /// Returns the size of the encrypted message.
    pub fn size(&self) -> f64 {
        self.size
    }

    /// Returns the current noise level in the ciphertext.
    pub fn noise(&self) -> f64 {
        self.noise
    }

    /// Decrypts the ciphertext using the provided private key. Ciphertext
    /// corruption can happen if the modulus is so large that the FFT multiplication
    /// precision is lost. This can lead to incorrect decryption results, in which case
    /// Err(CiphertextError::DecryptionFailed) will be returned.
    ///
    /// # Arguments
    /// * `private_key` - The private key used for decryption
    ///
    /// # Returns
    /// * `Ok([Complex<f64>; N / 2])` - The decrypted message as an array of complex numbers
    /// * `Err(CiphertextError)` - If the ciphertext is corrupted or decryption fails
    pub fn decrypt(
        &self,
        private_key: &PrivateKey,
    ) -> Result<[Complex<f64>; N / 2], CiphertextError> {
        let bits = self.bits();
        let mut s = private_key.secret().clone();
        s.change_modulus(bits);
        let c1 = &self.c1;
        let c2 = &self.c2;
        let poly = c1 + &(c2 * &s);
        let result = poly.slots();
        match result {
            Ok(slots) => {
                if self.scale_exp >= 32 {
                    return Err(CiphertextError::ScaleTooLarge);
                }
                let scale_as_float = (1 << self.scale_exp) as f64;
                Ok(std::array::from_fn(|i| slots[i] / scale_as_float))
            }
            Err(_) => Err(CiphertextError::CorruptedCiphertext),
        }
    }

    /// Rescales the ciphertext down to a smaller modulus. This
    /// divides the message by 2^bits.
    ///
    /// This operation is used for rescaling after multiplication.
    ///
    /// # Arguments
    /// * `bits` - The number of bits to decrease the modulus by
    pub fn scale_down(&mut self, bits: usize) {
        debug_assert!(
            self.bits > bits,
            "Cannot rescale by more than the current modulus"
        );
        self.bits -= bits;
        self.scale_exp -= bits;
        self.c1.decrease_precision(bits);
        self.c2.decrease_precision(bits);
    }

    /// Reduces the number of bits in a ciphertext. The modulus decreases
    /// by the specified number of bits, while the residues remain the same
    /// (up to the new modulus). This function is used to make the ciphertexts
    /// match bit counts when they are used in operations.
    ///
    /// # Arguments
    /// * `bits` - The number of bits to reduce the ciphertext by
    pub fn reduce_bits(&mut self, bits: usize) {
        debug_assert!(
            self.bits > bits,
            "Cannot reduce by more than the current modulus"
        );
        self.bits -= bits;
        self.c1.change_modulus(self.bits);
        self.c2.change_modulus(self.bits);
    }

    /// Moves the coefficients of the encrypted polynomial into its slots
    /// via an FFT.
    fn coefficients_to_slots(&self) -> Self {
        unimplemented!()
    }

    /// Moves the values of the slots into the coefficients of the encrypted polynomial
    /// via an inverse FFT.
    fn slots_to_coefficients(&self) -> Self {
        unimplemented!()
    }

    /// Refreshes the bit precision of the ciphertext. This is known as
    /// "bootstrapping" in the original CKKS scheme, although this step is
    /// not actually bootstrapping.
    ///
    /// This operation is used to restore the number of bits in the ciphertext
    /// after multiplication operations consume too much of it.
    fn refresh_bits(&self) {
        unimplemented!()
    }

    /// Cleans a ciphertext encrypting 0 or 1 by moving the stored value closer to
    /// 0 or 1.
    fn bit_clean(&self) {
        unimplemented!()
    }

    /// Cleans the ciphertext encrypting one of 0, 1, ..., k to the nearest integer.
    fn integer_clean(&self) {
        unimplemented!()
    }
}

/// Implementation of homomorphic addition for ciphertexts.
///
/// This allows two ciphertexts to be added together without decryption,
/// resulting in a ciphertext that encrypts the sum of the original plaintexts.
impl<'a> Add for &Ciphertext<'a> {
    type Output = Result<Ciphertext<'a>, CiphertextError>;

    /// Performs homomorphic addition of two ciphertexts. If the ciphertexts are not
    /// encrypted with the same keys, an error is returned. If the ciphertexts have different
    /// numbers of bits, then an error is returned. If the ciphertexts have different scales,
    /// then an error is returned.
    fn add(self, other: &Ciphertext<'a>) -> Result<Ciphertext<'a>, CiphertextError> {
        if !std::ptr::eq(self.keys(), other.keys()) {
            return Err(CiphertextError::KeyMismatch);
        }
        if (self.bits != other.bits) {
            return Err(CiphertextError::BitMismatch);
        }
        if (self.scale_exp != other.scale_exp) {
            return Err(CiphertextError::ScaleMismatch);
        }
        let keys = self.keys();
        let c1 = &self.c1 + &other.c1;
        let c2 = &self.c2 + &other.c2;
        let size = self.size() + other.size();
        let noise = self.noise() + other.noise();
        Ok(Ciphertext {
            scale_exp: self.scale_exp,
            bits: self.bits,
            c1,
            c2,
            keys,
            size,
            noise,
        })
    }
}

/// Implementation of homomorphic subtraction for ciphertexts.
///
/// This allows one ciphertext to be subtracted from another without decryption,
/// resulting in a ciphertext that encrypts the difference of the original plaintexts.
impl<'a> Sub for &Ciphertext<'a> {
    type Output = Result<Ciphertext<'a>, CiphertextError>;

    /// Performs homomorphic subtraction of two ciphertexts.
    ///
    /// # Panics
    /// Panics if the ciphertexts have different key packages.
    fn sub(self, other: &Ciphertext<'a>) -> Result<Ciphertext<'a>, CiphertextError> {
        if !std::ptr::eq(self.keys(), other.keys()) {
            return Err(CiphertextError::KeyMismatch);
        }
        self + &(-other)
    }
}

/// Implementation of homomorphic multiplication for ciphertexts.
///
/// This allows two ciphertexts to be multiplied together without decryption,
/// resulting in a ciphertext that encrypts the product of the original plaintexts.
/// This is the most complex homomorphic operation and requires relinearization.
impl<'a> Mul for &Ciphertext<'a> {
    type Output = Result<Ciphertext<'a>, CiphertextError>;

    /// Performs homomorphic multiplication of two ciphertexts.
    ///
    /// The multiplication involves:
    /// 1. Computing the product of polynomial components
    /// 2. Relinearization using evaluation keys to reduce ciphertext size
    /// 3. Rescaling to maintain precision
    ///
    /// If the ciphertexts have different key packages or bit precisions,
    /// an error is returned.
    /// If the ciphertexts have different scale exponents, the output
    /// will have a scale exponent equal to the minimum of the two.
    fn mul(self, other: &Ciphertext<'a>) -> Result<Ciphertext<'a>, CiphertextError> {
        if !std::ptr::eq(self.keys(), other.keys()) {
            return Err(CiphertextError::KeyMismatch);
        }
        if self.bits != other.bits {
            return Err(CiphertextError::BitMismatch);
        }
        let keys = self.keys();
        let evaluation_key = keys.evaluation_key();
        let ek1 = evaluation_key.k1();
        let ek2 = evaluation_key.k2();
        let p_exp = keys.p_exp();
        let bits = self.bits;
        let scale_exp = self.scale_exp + other.scale_exp;

        let c1p = self.c1() * other.c1();
        let c2p = &(self.c1() * other.c2()) + &(self.c2() * other.c1());
        let mut c3p = self.c2() * other.c2();
        c3p.change_modulus(bits + p_exp);
        let mut c3p_times_ek1_over_p = &c3p * ek1;
        let mut c3p_times_ek2_over_p = &c3p * ek2;
        c3p_times_ek1_over_p.decrease_precision(p_exp);
        c3p_times_ek2_over_p.decrease_precision(p_exp);

        let c1 = &c1p + &c3p_times_ek1_over_p;
        let c2 = &c2p + &c3p_times_ek2_over_p;

        let mut result = Ciphertext {
            scale_exp,
            bits: bits,
            c1,
            c2,
            keys: self.keys(),
            size: self.size(),
            noise: self.noise(),
        };
        let bits_to_reduce = std::cmp::max(self.scale_exp, other.scale_exp);
        result.scale_down(bits_to_reduce);
        Ok(result)
    }
}

/// Implementation of homomorphic negation for ciphertexts.
///
/// This allows a ciphertext to be negated without decryption,
/// resulting in a ciphertext that encrypts the negation of the original plaintext.
impl<'a> Neg for &Ciphertext<'a> {
    type Output = Ciphertext<'a>;

    /// Performs homomorphic negation of a ciphertext.
    fn neg(self) -> Ciphertext<'a> {
        let keys = self.keys();
        let c1 = -&self.c1;
        let c2 = -&self.c2;
        let size = self.size();
        let noise = self.noise();
        Ciphertext {
            scale_exp: self.scale_exp,
            bits: self.bits,
            c1,
            c2,
            keys,
            size,
            noise,
        }
    }
}
