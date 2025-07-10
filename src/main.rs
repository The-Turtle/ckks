use ckks::ciphertext::Ciphertext;
use ckks::key::{KeyPackage, PrivateKey};
use num_complex::Complex;

fn main() {
    let message1 = std::array::from_fn(|i| {
        if i <= 10 {
            Complex::<f64>::new(i as f64, (i * i) as f64)
        } else {
            Complex::<f64>::new(0.0, 0.0)
        }
    });
    let message2 = std::array::from_fn(|i| {
        if i <= 10 {
            Complex::<f64>::new(-(i as f64), -2.0 * (i as f64))
        } else {
            Complex::<f64>::new(0.0, 0.0)
        }
    });
    let noise = 1.0;
    let scale_exp = 30;

    let private_key = PrivateKey::new(1000, noise);
    let key_package = KeyPackage::new(&private_key);
    let ciphertext1 = Ciphertext::<'_>::from_message(message1, scale_exp, &key_package, noise);
    let ciphertext2 = Ciphertext::<'_>::from_message(message2, scale_exp, &key_package, noise);

    let product = &ciphertext1 * &ciphertext2;
    let decrypted_message = product.decrypt(&private_key);
    for val in decrypted_message.iter().take(10) {
        println!("{:?}", val);
    }
}
