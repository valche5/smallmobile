//! Un générateur rapide pour le laboratoire (SplitMix64). Sans rapport avec les dés du jeu.

#[derive(Clone, Copy, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x5851_f42d_4c95_7f2d)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Un entier dans [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
    }

    /// Un réel dans [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }

    /// Un générateur indépendant, dérivé de celui-ci.
    pub fn fork(&mut self) -> Rng {
        Rng(self.next_u64())
    }
}

/// Mélange deux nombres en une graine (pour dériver des graines reproductibles).
pub fn mix(a: u64, b: u64) -> u64 {
    Rng::new(a.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ b).next_u64()
}
