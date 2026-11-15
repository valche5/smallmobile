import { useState } from 'preact/hooks';
import { POWERS, RACES, startingTokens } from '../../engine/data';
import { pickCost } from '../../engine/rules';
import type { GameState } from '../../engine/types';
import { KIND_CLASS } from '../components/TopBar';
import { ArtIcon } from '../components/ArtIcon';
import { act } from '../store';

/** Le marché : chaque combinaison en 3 lignes, le prix et le bouton « Prendre » quand c'est à moi de choisir. */
export function MarketView({ g, human }: { g: GameState; human: boolean }) {
  const canPick = human && g.phase === 'pick';
  const p = g.players[g.current]!;
  const [open, setOpen] = useState<number | null>(null);
  const [help, setHelp] = useState(false);

  return (
    <div class="view scroll market">
      <div class="market-head">
        <h2>{canPick ? 'Choisissez un peuple' : 'Peuples disponibles'}</h2>
        <button class="help-btn" aria-label="Règle du prix" aria-expanded={help} onClick={() => setHelp(!help)}>
          ?
        </button>
      </div>
      {help && (
        <p class="market-help" onClick={() => setHelp(false)}>
          Le 1er peuple est gratuit. Plus bas, 1 💰 par peuple sauté{g.market.some((m) => m.power === 'cursed') ? ' (3 💰 pour des Maudits)' : ''} ; vous
          ramassez les 💰 posées dessus.
        </p>
      )}
      <div class="combo-list">
        {g.market.map((slot, i) => {
          const race = RACES[slot.race];
          const power = POWERS[slot.power];
          // jetons de départ : Homoncules ajoutés au marché, +2 de la Horde
          const tokens = startingTokens(slot);
          const cost = pickCost(g, i);
          // Historiens : +1 pièce par peuple en déclin à la table, dès le choix
          const historian = slot.power === 'historian' ? Object.values(g.races).filter((r) => r?.declined).length : 0;
          const net = slot.coins + historian - cost;
          const affordable = p.coins >= cost;
          const expanded = open === i;
          const price = `${cost === 0 ? 'Gratuit' : `−${cost} 💰`}${slot.coins ? ` · +${slot.coins} posées` : ''}${historian ? ` · +${historian} ${POWERS.historian.icon}` : ''}`;
          return (
            <div key={`${slot.race}-${slot.power}`} class={`combo-card ${expanded ? 'open' : ''} ${canPick && !affordable ? 'locked' : ''}`}>
              <button class="combo-main" onClick={() => setOpen(expanded ? null : i)} aria-expanded={expanded}>
                <div class="combo-l1">
                  <span class="combo-art">
                    <ArtIcon race={slot.race} size={32} />
                    <ArtIcon power={slot.power} size={28} />
                  </span>
                  <b class="combo-name">
                    {race.name} {power.name}
                  </b>
                  <span class="combo-tokens">
                    {tokens} jetons{slot.tokens ? ` (+${slot.tokens} ${race.icon})` : ''}
                  </span>
                </div>
                <div class={`combo-eff k-${KIND_CLASS[race.kind]}`}>
                  <ArtIcon race={slot.race} size={20} /> {race.short}
                </div>
                <div class="combo-l3">
                  <div class={`combo-eff k-${KIND_CLASS[power.kind]}`}>
                    <ArtIcon power={slot.power} size={20} /> {power.short}
                  </div>
                  <span class={`price ${net > 0 ? 'good' : ''}`}>{price}</span>
                </div>
                {expanded && (
                  <div class="combo-details">
                    <p>
                      <ArtIcon race={slot.race} /> <b>{race.name} :</b> {race.desc}
                    </p>
                    <p>
                      <ArtIcon power={slot.power} /> <b>{power.name} :</b> {power.desc}
                    </p>
                  </div>
                )}
              </button>
              {canPick && (
                <button class="btn primary take" disabled={!affordable} onClick={() => act({ type: 'pick', index: i })}>
                  {!affordable ? `Il vous faut ${cost} 💰` : net > 0 ? `Prendre · +${net} 💰` : net < 0 ? `Prendre · −${-net} 💰` : 'Prendre · gratuit'}
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
