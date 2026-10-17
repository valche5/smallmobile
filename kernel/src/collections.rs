//! Petites collections de taille fixe : l'état de partie reste `Copy`, sans allocation, pour être
//! copié des milliers de fois par les simulations.

use core::ops::{Deref, DerefMut};

/// Une liste d'au plus `N` éléments, rangée dans un tableau.
#[derive(Clone, Copy)]
pub struct ArrayVec<T: Copy + Default, const N: usize> {
    items: [T; N],
    len: u8,
}

/// Plus de place dans une `ArrayVec` : l'état décrit serait impossible dans une vraie partie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Full;

/// Deux listes sont égales si leurs éléments le sont (les cases libres ne comptent pas).
impl<T: Copy + Default + PartialEq, const N: usize> PartialEq for ArrayVec<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Default + Eq, const N: usize> Eq for ArrayVec<T, N> {}

impl<T: Copy + Default + core::fmt::Debug, const N: usize> core::fmt::Debug for ArrayVec<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T: Copy + Default, const N: usize> Default for ArrayVec<T, N> {
    fn default() -> Self {
        Self { items: [T::default(); N], len: 0 }
    }
}

impl<T: Copy + Default, const N: usize> ArrayVec<T, N> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn try_push(&mut self, x: T) -> Result<(), Full> {
        let len = self.len as usize;
        if len == N {
            return Err(Full);
        }
        self.items[len] = x;
        self.len += 1;
        Ok(())
    }

    /// Ajoute `x` ; les règles garantissent la place (dépasser est un bogue du noyau).
    pub fn push(&mut self, x: T) {
        self.try_push(x).expect("ArrayVec pleine")
    }

    pub fn remove(&mut self, at: usize) -> T {
        let len = self.len as usize;
        assert!(at < len);
        let x = self.items[at];
        self.items.copy_within(at + 1..len, at);
        self.len -= 1;
        x
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let mut n = 0;
        for i in 0..self.len as usize {
            let x = self.items[i];
            if keep(&x) {
                self.items[n] = x;
                n += 1;
            }
        }
        self.len = n as u8;
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn as_slice(&self) -> &[T] {
        &self.items[..self.len as usize]
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.items[..self.len as usize]
    }
}

impl<T: Copy + Default, const N: usize> Deref for ArrayVec<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy + Default, const N: usize> DerefMut for ArrayVec<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy + Default, const N: usize> IntoIterator for ArrayVec<T, N> {
    type Item = T;
    type IntoIter = core::iter::Take<core::array::IntoIter<T, N>>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter().take(self.len as usize)
    }
}

impl<'a, T: Copy + Default, const N: usize> IntoIterator for &'a ArrayVec<T, N> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<T: Copy + Default, const N: usize> FromIterator<T> for ArrayVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut out = Self::new();
        for x in iter {
            out.push(x);
        }
        out
    }
}

/// Un ensemble de régions (au plus 64), sous forme de bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegionSet(pub u64);

impl RegionSet {
    pub const EMPTY: RegionSet = RegionSet(0);

    pub fn contains(self, i: usize) -> bool {
        self.0 & (1 << i) != 0
    }

    pub fn insert(&mut self, i: usize) {
        self.0 |= 1 << i;
    }

    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// La plus petite région de l'ensemble.
    pub fn first(self) -> Option<usize> {
        (self.0 != 0).then(|| self.0.trailing_zeros() as usize)
    }

    pub fn intersects(self, other: RegionSet) -> bool {
        self.0 & other.0 != 0
    }

    /// Les régions, dans l'ordre croissant.
    pub fn iter(self) -> impl Iterator<Item = usize> {
        let mut bits = self.0;
        core::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            Some(i)
        })
    }
}

impl FromIterator<usize> for RegionSet {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut s = RegionSet::EMPTY;
        for i in iter {
            s.insert(i);
        }
        s
    }
}

/// Un ensemble de joueurs (au plus 8), sous forme de bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlayerSet(pub u8);

impl PlayerSet {
    pub fn contains(self, p: usize) -> bool {
        self.0 & (1 << p) != 0
    }

    pub fn insert(&mut self, p: usize) {
        self.0 |= 1 << p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_vec_keeps_order() {
        let mut v: ArrayVec<u8, 4> = [1, 2, 3].into_iter().collect();
        assert_eq!(v.remove(0), 1);
        v.push(4);
        v.retain(|&x| x != 3);
        assert_eq!(v.as_slice(), &[2, 4]);
        assert_eq!(v.try_push(5), Ok(()));
        assert_eq!(v.try_push(6), Ok(()));
        assert_eq!(v.try_push(7), Err(Full));
    }

    #[test]
    fn region_set_iterates_in_order() {
        let s: RegionSet = [40, 3, 17].into_iter().collect();
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![3, 17, 40]);
        assert_eq!(s.first(), Some(3));
        assert_eq!(s.len(), 3);
    }
}
