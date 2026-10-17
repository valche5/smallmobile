/** Une graine au hasard, pour une nouvelle partie ou une nouvelle carte. */
export function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 31);
}
