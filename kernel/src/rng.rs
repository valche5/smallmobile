//! Le hasard, déterministe : les dés (mulberry32) et le bruit de l'IA (FNV-1a sur un texte).

use crate::ids::DIE_FACES;

/// Un tirage dans [0, 1) à partir de l'état `state` ; renvoie aussi l'état suivant.
pub fn step(state: u32) -> (f64, u32) {
    let next = state.wrapping_add(0x6d2b_79f5);
    let mut t = (next ^ (next >> 15)).wrapping_mul(next | 1);
    t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
    ((t ^ (t >> 14)) as f64 / 4_294_967_296.0, next)
}

/// Premier tirage d'un générateur initialisé avec `seed`.
pub fn draw(seed: u32) -> f64 {
    step(seed).0
}

/// Lance le dé de renfort.
pub fn roll_die(rng: &mut u32) -> i32 {
    let (v, next) = step(*rng);
    *rng = next;
    DIE_FACES[(v * DIE_FACES.len() as f64) as usize]
}

/// FNV-1a sur le texte d'une clé du type `"12:345:pick3"`, sans construire le texte.
#[derive(Clone, Copy)]
pub struct Key(u32);

impl Key {
    pub fn new() -> Key {
        Key(2_166_136_261)
    }

    fn byte(mut self, b: u8) -> Key {
        self.0 = (self.0 ^ b as u32).wrapping_mul(16_777_619);
        self
    }

    pub fn text(self, s: &str) -> Key {
        s.bytes().fold(self, Key::byte)
    }

    /// Un entier écrit en décimal.
    pub fn num(self, n: u64) -> Key {
        let mut digits = [0u8; 20];
        let mut at = digits.len();
        let mut n = n;
        loop {
            at -= 1;
            digits[at] = b'0' + (n % 10) as u8;
            n /= 10;
            if n == 0 {
                break;
            }
        }
        digits[at..].iter().fold(self, |k, &b| k.byte(b))
    }

    pub fn sep(self) -> Key {
        self.byte(b':')
    }

    pub fn finish(self) -> u32 {
        self.0
    }
}

impl Default for Key {
    fn default() -> Self {
        Key::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_typescript_generator() {
        // mulberry32(42)() et hashString("12:345:pick3") en TypeScript
        assert_eq!(draw(42), 0.6011037519201636);
        assert_eq!(Key::new().num(12).sep().num(345).sep().text("pick").num(3).finish(), 658_152_922);
    }
}
