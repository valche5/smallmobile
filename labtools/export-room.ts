// Exporte une partie en ligne sauvegardée par le serveur (data/rooms/CODE.json) au format du noyau,
// pour que le labo l'analyse :
//
//   bun labtools/export-room.ts data/rooms/ABCD.json /tmp/ABCD.smk
//   ./target/release/lab1 analyze --game /tmp/ABCD.smk --agent az:net=lab1/nets/gen11.bin,n=20000
import { readFileSync, writeFileSync } from 'node:fs';
import { gameWords } from '../src/engine/kernel';

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error('usage : bun labtools/export-room.ts data/rooms/CODE.json sortie.smk');
  process.exit(2);
}
const room = JSON.parse(readFileSync(input, 'utf8'));
if (!room.game) {
  console.error("Cette partie n'a pas encore commencé.");
  process.exit(1);
}
const words = gameWords(room.game);
writeFileSync(output, new Uint8Array(words.buffer, words.byteOffset, words.byteLength));
console.log(`${output} : ${words.length} mots, manche ${room.game.round}`);
