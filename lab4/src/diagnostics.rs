//! Résumés et export des compteurs de recherche. Les durées des décisions restent dans l'arène.

use std::io::{self, Write};

use crate::az::Az;
use smallmobile_brain4::diagnostics::Diagnostics;

pub fn describe(d: Diagnostics) -> String {
    let pct = |n| 100.0 * n as f64 / d.simulations.max(1) as f64;
    format!(
        "{} recherches, {} simulations ; racines : {}/{} coups explorés, {} explorés hors top16, \
         {} choix hors top16 ({} racines avec >16 coups) ; profondeur moyenne {:.1}, max {} ; \
         tour courant terminé {:.1} %, prochain tour du joueur atteint {:.1} %, partie finie {:.1} %",
        d.searches,
        d.simulations,
        d.visited,
        d.legal,
        d.visited_outside16,
        d.chosen_outside16,
        d.wide_roots,
        d.depth_sum as f64 / d.simulations.max(1) as f64,
        d.max_depth,
        pct(d.turn_finished),
        pct(d.next_turn),
        pct(d.game_over),
    )
}

pub fn print(az: &Az) {
    for (i, d) in az.diagnostics().into_iter().enumerate().filter(|(_, d)| d.searches > 0) {
        println!("recherche {}, {} joueurs : {}", az.describe(), i + 2, describe(d));
    }
}

/// Une ligne par rôle et taille, compteurs bruts pour agréger sans moyenner des pourcentages.
pub fn write(out: &mut impl Write, seed: u64, agents: &[(usize, &Az)]) -> io::Result<()> {
    writeln!(out, "# graine {seed}")?;
    writeln!(
        out,
        "role\tplayers\tagent\tsearches\tsimulations\tlegal\tvisited\twide_roots\tvisited_outside16\tchosen_outside16\tdepth_sum\tmax_depth\tturn_finished\tnext_turn\tgame_over"
    )?;
    for &(role, az) in agents {
        for (i, d) in az.diagnostics().into_iter().enumerate().filter(|(_, d)| d.searches > 0) {
            writeln!(
                out,
                "{role}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                i + 2,
                az.describe(),
                d.searches,
                d.simulations,
                d.legal,
                d.visited,
                d.wide_roots,
                d.visited_outside16,
                d.chosen_outside16,
                d.depth_sum,
                d.max_depth,
                d.turn_finished,
                d.next_turn,
                d.game_over
            )?;
        }
    }
    out.flush()
}
