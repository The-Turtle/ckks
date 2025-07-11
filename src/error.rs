use std::fmt;

/// Errors that can occur during ciphertext operations in the CKKS homomorphic encryption scheme.
#[derive(Debug)]
pub enum CiphertextError {
    /// The ciphertext data has been corrupted. This is usually due to
    /// loss of precision during ciphertext multiplication from FFT.
    CorruptedCiphertext,

    /// The operation requires ciphertexts that were encrypted with the same key,
    /// but the provided ciphertexts use different keys.
    KeyMismatch,

    /// The ciphertexts have different scaling factors, which is incompatible
    /// for addition.
    ScaleMismatch,

    /// The ciphertexts have different bit precisions or were encrypted with
    /// different parameter sets that affect the bit representation.
    BitMismatch,

    /// The scaling factor is too large, causing the decryption
    /// of the messages to exceed the 64-bit threshold.
    ScaleTooLarge,
}

impl fmt::Display for CiphertextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CiphertextError::CorruptedCiphertext => write!(f, "Corrupted ciphertext"),
            CiphertextError::KeyMismatch => write!(f, "Key mismatch"),
            CiphertextError::ScaleMismatch => write!(f, "Scale mismatch"),
            CiphertextError::BitMismatch => write!(f, "Bit mismatch"),
            CiphertextError::ScaleTooLarge => write!(f, "Scale too large"),
        }
    }
}

/// Errors that can occur during arithmetic operations in the CKKS scheme.
///
/// These errors represent mathematical operation failures that can occur
/// during homomorphic computations or related arithmetic operations.
#[derive(Debug)]
pub enum ArithmeticError {
    /// An arithmetic overflow occurred during computation.
    /// This occurs when BigInts are converted to u64 and exceed the 64-bit threshold.
    Overflow,
}

impl fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArithmeticError::Overflow => write!(f, "Overflow"),
        }
    }
}
