use ckks::ciphertext::Ciphertext;
use ckks::key::{KeyPackage, PrivateKey};
use num_complex::Complex;

fn main() {
    let message1 = Complex::<f64>::new(5.0, 0.0);
    let message2 = Complex::<f64>::new(6.0, 0.0);
    let message3 = Complex::<f64>::new(7.0, 0.0);
    let noise = 1.0;
    let scale_exp = 20;

    let private_key = PrivateKey::new(1000, noise);
    let key_package = KeyPackage::new(&private_key);
    let ciphertext1 = Ciphertext::<'_>::from_complex(message1, scale_exp, &key_package, noise);
    let ciphertext2 = Ciphertext::<'_>::from_complex(message2, scale_exp, &key_package, noise);
    let mut ciphertext3 = Ciphertext::<'_>::from_complex(message3, scale_exp, &key_package, noise);

    let product = (&ciphertext1 * &ciphertext2).expect("Multiplication failed");
    ciphertext3.reduce_bits(scale_exp);
    let sum = (&product + &ciphertext3).expect("Addition failed");
    let result = sum.decrypt(&private_key);
    match result {
        Ok(decrypted_message) => {
            for val in decrypted_message.iter().take(10) {
                println!("{:?}", val);
            }
        }
        Err(err) => println!("Error: {}", err),
    }
}
