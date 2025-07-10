use crate::ring::Poly;

pub struct Key {
    k1: Poly,
    k2: Poly,
    bits: usize,
}

impl Key {
    fn from_polys(k1: Poly, k2: Poly) -> Self {
        debug_assert!(k1.bits() == k2.bits());
        let bits = k1.bits();
        Key { k1, k2, bits }
    }

    pub fn k1(&self) -> &Poly {
        &self.k1
    }

    pub fn k2(&self) -> &Poly {
        &self.k2
    }

    pub fn change_modulus(&mut self, new_modulus: usize) {
        self.k1.change_modulus(new_modulus);
        self.k2.change_modulus(new_modulus);
    }

    fn rotated(&self, n: i64) -> Self {
        debug_assert!(n % 4 == 1);
        let k1 = self.k1.substitution(n);
        let k2 = self.k2.substitution(n);
        Key::from_polys(k1, k2)
    }

    fn conjugated(&self) -> Self {
        let k1 = self.k1.substitution(-1);
        let k2 = self.k2.substitution(-1);
        Key::from_polys(k1, k2)
    }
}

pub struct PrivateKey {
    key: Key,
    noise: f64,
}

impl PrivateKey {
    pub fn new(bits: usize, noise: f64) -> Self {
        let k1 = Poly::from_int(bits, 1);
        let k2 = Poly::random_small(bits, noise);
        let key = Key::from_polys(k1, k2);
        PrivateKey { key, noise }
    }

    pub fn bits(&self) -> usize {
        self.key.bits
    }

    pub fn noise(&self) -> f64 {
        self.noise
    }

    pub fn secret(&self) -> &Poly {
        &self.key.k2
    }
}

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
    pub fn new(private_key: &PrivateKey) -> Self {
        let bits = private_key.bits();
        let s = private_key.secret();

        let noise = private_key.noise();

        let a = Poly::random(bits);
        let error = Poly::random_small(bits, noise);
        let public_k1 = &(&-&a * s) + &error;
        let public_k2 = a;
        let public_key = Key::from_polys(public_k1, public_k2);

        let p_exp = bits;

        let a = Poly::random(bits + p_exp);
        let error = Poly::random_small(bits + p_exp, noise);
        let mut s_lifted = s.clone();
        s_lifted.change_modulus(bits + p_exp);
        let ek1 = &-&(&(&a * &s_lifted) + &error) + &(&(s * s) << p_exp);
        let ek2 = a;
        let evaluation_key = Key::from_polys(ek1, ek2);

        // TODO
        let rotation_keys = vec![Key::from_polys(
            Poly::from_int(bits, 1),
            Poly::from_int(bits, 1),
        )];
        let conjugation_key = Key::from_polys(Poly::from_int(bits, 1), Poly::from_int(bits, 1));

        let p_exp = bits;

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

    pub fn bits(&self) -> usize {
        self.bits
    }

    pub fn public_key(&self) -> &Key {
        &self.public_key
    }

    pub fn evaluation_key(&self) -> &Key {
        &self.evaluation_key
    }

    pub fn rotation_key(&self, index: usize) -> &Key {
        &self.rotation_keys[index]
    }

    pub fn conjugation_key(&self) -> &Key {
        &self.conjugation_key
    }

    pub fn noise(&self) -> f64 {
        self.noise
    }

    pub fn p_exp(&self) -> usize {
        self.p_exp
    }
}
