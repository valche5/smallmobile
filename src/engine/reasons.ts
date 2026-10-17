/** Les raisons de refus du noyau (`Reason::code`, kernel/src/reason.rs), en français. */
import type { Reason } from './kernel';
import type { GameState } from './types';

function plural(n: number, word: string): string {
  return `${n} ${word}${n > 1 ? 's' : ''}`;
}

export function reasonText(g: GameState, { code, arg }: Reason): string {
  const name = (pid: number) => g.players[pid]?.name ?? '?';
  switch (code) {
    case 1: return "Ce n'est pas le moment de choisir un peuple.";
    case 2: return "Ce n'est pas la phase de conquête.";
    case 3: return "Ce n'est pas la phase de redéploiement.";
    case 4: return "Ce n'est pas la fin du tour.";
    case 5: return "Il n'y a pas de redéploiement défensif en cours.";
    case 6: return 'Possible seulement pendant le redéploiement de votre peuple actif.';
    case 7: return 'La partie est terminée.';
    case 8: return "Ce peuple n'est pas disponible.";
    case 9: return `Il vous faut ${arg} pièces pour prendre ce peuple.`;
    case 10: return 'Des peuples sont disponibles : choisissez-en un.';
    case 11: return 'Vous occupez déjà cette région.';
    case 12: return 'Vous ne pouvez pas attaquer votre propre peuple actif.';
    case 13: return 'Vos Goules ne peuvent pas attaquer vos autres peuples.';
    case 14: return `Paix diplomatique : ${name(arg)} vous interdit d'attaquer son peuple actif ce tour-ci.`;
    case 15: return `${arg ? 'Les mers' : 'Les lacs'} ne peuvent pas être conquis (sauf par les Marins).`;
    case 16:
      return arg === 0
        ? 'Un héros protège cette région : elle est imprenable.'
        : arg === 1
          ? 'Un terrier de Semi-hommes protège cette région : elle est imprenable.'
          : 'Un dragon garde cette région : elle est imprenable.';
    case 17: return "Première conquête : choisissez une région en bord de carte (ou sur la côte d'une mer en bordure).";
    case 18: return "Cette région n'est pas adjacente à vos régions.";
    case 19: return 'Le dé de renfort a été lancé : vos conquêtes sont terminées.';
    case 20: return "Vous n'avez plus de jeton en main.";
    case 21: return `Il vous faut ${arg} jetons pour cette conquête.`;
    case 22: return 'Vous avez assez de jetons : conquérez normalement.';
    case 23: return 'Même avec le dé, cette conquête est impossible.';
    case 24: return 'Seuls les Berserks lancent le dé avant chaque conquête.';
    case 25: return 'Le dé a déjà été lancé pour cette conquête.';
    case 26: return 'Il faut au moins 1 jeton en main.';
    case 27: return 'Le dragon a déjà été utilisé ce tour.';
    case 28: return 'Le dragon ne peut pas attaquer cette région.';
    case 29: return 'La sorcellerie ne touche que les peuples actifs.';
    case 30: return 'La sorcellerie exige un jeton adverse isolé (exactement 1 jeton).';
    case 31: return `Vous avez déjà ensorcelé ${name(arg)} ce tour.`;
    case 32: return 'Plus aucun jeton Sorcier disponible.';
    case 33: return 'Sorcellerie impossible ici.';
    case 34: return 'Possible seulement au début de la phase de conquête.';
    case 35: return "Cette région n'est pas occupée par votre peuple en action.";
    case 36: return "On ne peut abandonner une région qu'avant la première conquête du tour.";
    case 37: return 'Le déclin se décide au début du tour.';
    case 38: return "Terminez d'abord le tour de vos Goules.";
    case 39: return "Vous n'avez pas de peuple actif.";
    case 40: return 'Impossible le tour où vous choisissez un peuple.';
    case 41: return 'Le déclin doit être décidé avant toute conquête.';
    case 42: return "Cette région n'est pas à vous.";
    case 43: return 'Une région occupée garde au moins 1 jeton.';
    case 44: return "Vous n'avez pas assez de jetons.";
    case 45: return 'Votre peuple actif ne possède pas ce pouvoir.';
    case 46: return 'Vos 5 campements sont déjà posés.';
    case 47: return 'Pas de campement ici.';
    case 48: return 'Une seule forteresse par tour.';
    case 49: return 'Il y a déjà une forteresse ici.';
    case 50: return 'Les 6 forteresses sont déjà construites.';
    case 51: return "Vos 2 héros sont déjà placés : retirez-en un d'abord.";
    case 52: return `Placez d'abord vos ${plural(arg, 'jeton')} en main.`;
    case 53: return 'Seuls les Diplomates peuvent imposer la paix.';
    case 54: return 'Adversaire invalide.';
    case 55: return 'Vous avez attaqué ce joueur ce tour : pas de paix possible.';
    case 56: return 'Seuls les Robustes peuvent décliner en fin de tour.';
    case 57: return 'Région inconnue.';
    default: return 'Action impossible.';
  }
}
