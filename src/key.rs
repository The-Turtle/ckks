use crate::ring::Poly;

/// A cryptographic key pair consisting of two polynomials in Z/(2^bits*Z)[X]/(X^N + 1).
/// This represents a general key structure used in CKKS encryption.
pub struct Key {
    k1: Poly,
    k2: Poly,
    bits: usize,
}

impl Key {
    /// Creates a new Key from two polynomials.
    /// Both polynomials must have the same bit size.
    fn from_polys(k1: Poly, k2: Poly) -> Self {
        debug_assert!(k1.bits() == k2.bits());
        let bits = k1.bits();
        Key { k1, k2, bits }
    }

    /// Returns a reference to the first polynomial of the key pair.
    pub fn k1(&self) -> &Poly {
        &self.k1
    }

    /// Returns a reference to the second polynomial of the key pair.
    pub fn k2(&self) -> &Poly {
        &self.k2
    }

    /// Changes the modulus of both polynomials in the key pair.
    /// This is used for modulus switching operations in CKKS.
    pub fn change_modulus(&mut self, new_modulus: usize) {
        self.k1.change_modulus(new_modulus);
        self.k2.change_modulus(new_modulus);
    }

    /// Creates a rotated version of the key by applying the substitution
    /// X -> X^n mod (X^N + 1).
    /// This is used for creating rotation keys in CKKS.
    fn rotated(&self, n: i64) -> Self {
        debug_assert!(n % 4 == 1);
        let k1 = self.k1.substitution(n);
        let k2 = self.k2.substitution(n);
        Key::from_polys(k1, k2)
    }

    /// Creates a conjugated version of the key by applying the substitution
    /// X -> X^-1 mod (X^N + 1).
    /// This is used for complex conjugation operations in CKKS.
    fn conjugated(&self) -> Self {
        let k1 = self.k1.substitution(-1);
        let k2 = self.k2.substitution(-1);
        Key::from_polys(k1, k2)
    }
}

/// The private key used for CKKS encryption/decryption.
/// Contains the secret key polynomial and noise parameter.
pub struct PrivateKey {
    key: Key,
    noise: f64,
}

impl PrivateKey {
    /// Creates a new private key with the specified bit size and noise level.
    /// k1 is set to the constant polynomial 1, k2 is a random small polynomial.
    pub fn new(bits: usize, noise: f64) -> Self {
        let k1 = Poly::from_int(bits, 1);
        let k2 = Poly::random_small(bits, noise);
        let key = Key::from_polys(k1, k2);
        PrivateKey { key, noise }
    }

    /// Returns the bit size of the private key.
    pub fn bits(&self) -> usize {
        self.key.bits
    }

    /// Returns the noise parameter used in key generation.
    pub fn noise(&self) -> f64 {
        self.noise
    }

    /// Returns a reference to the secret polynomial (k2).
    /// This is the actual secret key used for encryption/decryption.
    pub fn secret(&self) -> &Poly {
        &self.key.k2
    }
}

/// A complete set of keys needed for CKKS homomorphic encryption operations.
/// Includes public key, evaluation key (for multiplication), rotation keys
/// (for cyclic rotation of slots), and conjugation key (for conjugation of slots).
pub struct KeyPackage {
    bits: usize,
    public_key: Key,
    evaluation_key: Key,
    rotation_keys: Vec<Key>,
    conjugation_key: Key,
    noise: f64,
    p_exp: usize,
}

impl KeyPackage {
    /// Creates a new KeyPackage from a private key.
    /// Generates all necessary keys for CKKS operations including public key,
    /// evaluation key, rotation keys, and conjugation key.
    pub fn new(private_key: &PrivateKey) -> Self {
        let bits = private_key.bits();
        let s = private_key.secret();

        let noise = private_key.noise();

        // Generate public key: (pk0, pk1) = (-a*s + e, a) where a is random and e is small error
        let a = Poly::random(bits);
        let error = Poly::random_small(bits, noise);
        let public_k1 = &(&-&a * s) + &error;
        let public_k2 = a;
        let public_key = Key::from_polys(public_k1, public_k2);

        let p_exp = bits;

        // Generate evaluation key for homomorphic multiplication (relinearization)
        // evk = (-a*s + e + s^2*P, a) where P is a scaling factor
        let a = Poly::random(bits + p_exp);
        let error = Poly::random_small(bits + p_exp, noise);
        let mut s_lifted = s.clone();
        s_lifted.change_modulus(bits + p_exp);
        let ek1 = &-&(&(&a * &s_lifted) + &error) + &(&(s * s) << p_exp);
        let ek2 = a;
        let evaluation_key = Key::from_polys(ek1, ek2);

        // TODO: Generate proper rotation keys for different rotation amounts
        // Currently using placeholder keys
        let rotation_keys = vec![Key::from_polys(
            Poly::from_int(bits, 1),
            Poly::from_int(bits, 1),
        )];
        // TODO: Generate proper conjugation key for complex conjugation
        // Currently using placeholder key
        let conjugation_key = Key::from_polys(Poly::from_int(bits, 1), Poly::from_int(bits, 1));

        Self {
            bits,
            public_key,
            evaluation_key,
            rotation_keys,
            conjugation_key,
            noise,
            p_exp,
        }
    }

    /// Returns the bit size of the key package.
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// Returns a reference to the public key used for encryption.
    pub fn public_key(&self) -> &Key {
        &self.public_key
    }

    /// Returns a reference to the evaluation key used for relinearization after multiplication.
    pub fn evaluation_key(&self) -> &Key {
        &self.evaluation_key
    }

    /// Returns a reference to the rotation key at the specified index.
    /// Used for homomorphic rotation operations on encrypted vectors.
    pub fn rotation_key(&self, index: usize) -> &Key {
        &self.rotation_keys[index]
    }

    /// Returns a reference to the conjugation key used for complex conjugation operations.
    pub fn conjugation_key(&self) -> &Key {
        &self.conjugation_key
    }

    /// Returns the noise parameter used in key generation.
    pub fn noise(&self) -> f64 {
        self.noise
    }

    /// Returns the exponent used for the scaling factor P in evaluation key generation,
    /// rotation key generation, and conjugation key generation.
    pub fn p_exp(&self) -> usize {
        self.p_exp
    }
}
