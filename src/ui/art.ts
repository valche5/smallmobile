import amazons from '../assets/illustrations/race-amazons.webp';
import dwarves from '../assets/illustrations/race-dwarves.webp';
import elves from '../assets/illustrations/race-elves.webp';
import ghouls from '../assets/illustrations/race-ghouls.webp';
import giants from '../assets/illustrations/race-giants.webp';
import halflings from '../assets/illustrations/race-halflings.webp';
import humans from '../assets/illustrations/race-humans.webp';
import orcs from '../assets/illustrations/race-orcs.webp';
import ratmen from '../assets/illustrations/race-ratmen.webp';
import skeletons from '../assets/illustrations/race-skeletons.webp';
import sorcerers from '../assets/illustrations/race-sorcerers.webp';
import tritons from '../assets/illustrations/race-tritons.webp';
import trolls from '../assets/illustrations/race-trolls.webp';
import wizards from '../assets/illustrations/race-wizards.webp';
import alchemist from '../assets/illustrations/power-alchemist.webp';
import berserk from '../assets/illustrations/power-berserk.webp';
import bivouacking from '../assets/illustrations/power-bivouacking.webp';
import commando from '../assets/illustrations/power-commando.webp';
import diplomat from '../assets/illustrations/power-diplomat.webp';
import dragonMaster from '../assets/illustrations/power-dragonMaster.webp';
import flying from '../assets/illustrations/power-flying.webp';
import fortified from '../assets/illustrations/power-fortified.webp';
import heroic from '../assets/illustrations/power-heroic.webp';
import merchant from '../assets/illustrations/power-merchant.webp';
import mounted from '../assets/illustrations/power-mounted.webp';
import pillaging from '../assets/illustrations/power-pillaging.webp';
import seafaring from '../assets/illustrations/power-seafaring.webp';
import spirit from '../assets/illustrations/power-spirit.webp';
import stout from '../assets/illustrations/power-stout.webp';
import underworld from '../assets/illustrations/power-underworld.webp';
import wealthy from '../assets/illustrations/power-wealthy.webp';
import farmland from '../assets/illustrations/terrain-farmland.webp';
import forest from '../assets/illustrations/terrain-forest.webp';
import hill from '../assets/illustrations/terrain-hill.webp';
import mountain from '../assets/illustrations/terrain-mountain.webp';
import swamp from '../assets/illustrations/terrain-swamp.webp';
import sea from '../assets/illustrations/terrain-sea.webp';
import lake from '../assets/illustrations/terrain-lake.webp';
import farmlandTexture from '../assets/illustrations/texture-farmland.webp';
import forestTexture from '../assets/illustrations/texture-forest.webp';
import hillTexture from '../assets/illustrations/texture-hill.webp';
import mountainTexture from '../assets/illustrations/texture-mountain.webp';
import swampTexture from '../assets/illustrations/texture-swamp.webp';
import seaTexture from '../assets/illustrations/texture-sea.webp';
import lakeTexture from '../assets/illustrations/texture-lake.webp';
import type { PowerId, RaceId, Terrain } from '../engine/types';

/** Chaque peuple, pouvoir et terrain dispose d’une illustration. */
export const RACE_ART: Record<RaceId, string> = {
  amazons, dwarves, elves, ghouls, giants, halflings, humans,
  orcs, ratmen, skeletons, sorcerers, tritons, trolls, wizards,
};
export const POWER_ART: Record<PowerId, string> = {
  alchemist, berserk, bivouacking, commando, diplomat, dragonMaster, flying,
  forest, fortified, heroic, hill, merchant, mounted, pillaging, seafaring,
  spirit, stout, swamp, underworld, wealthy,
};
export const TERRAIN_ART: Record<Terrain, string> = {
  farmland, forest, hill, mountain, swamp, sea, lake,
};
export const TERRAIN_TEXTURES: Record<Terrain, string> = {
  farmland: farmlandTexture,
  forest: forestTexture,
  hill: hillTexture,
  mountain: mountainTexture,
  swamp: swampTexture,
  sea: seaTexture,
  lake: lakeTexture,
};
