//! Les options de lab8 : sans elles, la recherche est celle de lab6 au bit près ; avec elles, les
//! recherches groupées (le chemin du GPU) rendent les mêmes coups que seules, et des parties
//! entières se jouent jusqu'au bout.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use smallmobile_kernel::{Game, Phase};
use smallmobile_lab8::agent::Agent;
use smallmobile_lab8::arena::play_game;
use smallmobile_lab8::az::{Az, Config, Evals, Searched};
use smallmobile_lab8::batch::Batch;
use smallmobile_lab8::know::Knowledge;
use smallmobile_lab8::moves;
use smallmobile_lab8::net::Net;
use smallmobile_lab8::rand::Rng;
use smallmobile_lab8::table::Table;

fn net(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("nets/{name}.bin"))
}

fn az(spec: &str) -> Az {
    let cfg = Config::parse(spec).expect("réglages");
    Az::new(cfg, Arc::new(Net::load(&net("c130")).expect("c130")), "c130".into())
}

/// Une position sur `every` où le joueur qui agit a le choix, dans une partie jouée au hasard.
fn positions(table: &Table, seed: u32, every: usize, most: usize) -> Vec<(Game<'_>, Knowledge)> {
    let (mut g, mut k) = (table.game(seed), Knowledge::default());
    let (mut rng, mut out, mut step) = (Rng::new(seed as u64), Vec::new(), 0);
    while g.state.phase != Phase::GameOver && out.len() < most {
        let legal = moves::legal(&g, &k);
        if legal.len() > 1 {
            if step % every == 0 {
                out.push((g, k));
            }
            step += 1;
        }
        k.play(&mut g, &legal[rng.below(legal.len())].mv).expect("coup possible");
    }
    out
}

/// Sans option (ou avec leurs valeurs par défaut écrites), la recherche de lab8 est celle de lab6 :
/// mêmes coups, politique, visites, valeurs, et même hasard consommé.
#[test]
fn without_options_the_search_is_lab6s() {
    use smallmobile_brain6 as old;
    let old_az = old::search::Az::new(
        old::search::Config { iterations: 64 },
        Arc::new(old::net::Net::load(&net("c130")).expect("c130")),
        "c130".into(),
    );
    for spec in ["n=64", "n=64,win=0.5,roll=0,mix=1,market=now,crn=0,explore=0,history=1"] {
        let new_az = az(spec);
        for n in 2..=5 {
            let table = Table::new(n, 73 + n as u64);
            let (mut game, mut k, mut before) = (table.game(91), Knowledge::default(), old::know::Knowledge::default());
            let mut random = Rng::new(141);
            for step in 0..200 {
                if game.state.phase == Phase::GameOver {
                    break;
                }
                let legal = moves::legal(&game, &k);
                if step % 23 == 0 && legal.len() > 1 {
                    for explore in [false, true] {
                        let (mut r1, mut r2) = (Rng::new(17), old::rand::Rng::new(17));
                        let a = new_az.search(&game, &k, &mut r1, explore, &mut Evals::default());
                        let b = old_az.search(&game, &before, &mut r2, explore, &mut old::search::Evals::default());
                        assert_eq!(format!("{:?}", a.mv), format!("{:?}", b.mv));
                        assert_eq!(a.policy, b.policy);
                        assert_eq!(a.visits, b.visits);
                        assert_eq!(a.finals.map(f32::to_bits), b.finals.map(f32::to_bits));
                        assert_eq!(a.wins.map(f32::to_bits), b.wins.map(f32::to_bits));
                        assert_eq!(a.value.to_bits(), b.value.to_bits());
                        assert_eq!(r1.next_u64(), r2.next_u64());
                        if !explore {
                            assert_eq!(a.greedy, a.mv, "sans bruit, le coup joué est le coup sans bruit");
                        }
                    }
                }
                let choice = legal[random.below(legal.len())];
                let old_choice = old::moves::legal(&game, &before).into_iter().find(|c| c.key == choice.key).unwrap();
                before.play(&mut game.clone(), &old_choice.mv).unwrap();
                k.play(&mut game, &choice.mv).unwrap();
            }
        }
    }
}

/// Le résultat d'une recherche, comparable au bit près.
fn print(s: &Searched) -> String {
    let bits = |xs: &[f32]| xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    format!("{:?} {:?} {:?} {:?} {:?} {}", s.mv, s.policy, s.visits, bits(&s.finals), bits(&s.wins), s.value.to_bits())
}

/// Des recherches avec fins simulées, marché tardif et mondes communs, menées ensemble et évaluées
/// par groupes (`Batch` sur le processeur : le chemin du GPU, au calcul près) : mêmes résultats que
/// chacune seule, évaluation par évaluation (`Direct`). Les fins simulées d'un même tour du halving
/// attendent leurs évaluations en même temps, chacune la sienne.
#[test]
fn grouped_searches_with_rollouts_match_lone_ones() {
    let az = az("n=24,roll=1,market=late,crn=1,mix=0.5");
    let tables: Vec<Table> = (2..=5).map(|n| Table::new(n, 400 + n as u64)).collect();
    let all: Vec<(Game, Knowledge)> = tables.iter().flat_map(|t| positions(t, 9, 17, 3)).collect();
    let alone: Vec<String> =
        all.iter().map(|(g, k)| print(&az.search(g, k, &mut Rng::new(5), false, &mut Evals::default()))).collect();
    let batch = Batch::new(&az.net, None);
    let mut running: Vec<Option<Pin<Box<dyn Future<Output = Searched> + '_>>>> = all
        .iter()
        .map(|(g, k)| {
            let (az, batch) = (&az, &batch);
            let search: Pin<Box<dyn Future<Output = Searched> + '_>> = Box::pin(async move {
                let mut evals = Evals::default();
                az.run(g, k, &mut Rng::new(5), az.cfg.iterations, false, &mut evals, batch).await
            });
            Some(search)
        })
        .collect();
    let mut done: Vec<Option<String>> = vec![None; all.len()];
    let mut cx = Context::from_waker(Waker::noop());
    while running.iter().any(Option::is_some) {
        for (slot, out) in running.iter_mut().zip(&mut done) {
            if let Some(f) = slot
                && let Poll::Ready(s) = f.as_mut().poll(&mut cx)
            {
                *out = Some(print(&s));
                *slot = None;
            }
        }
        batch.submit();
    }
    for (a, b) in alone.iter().zip(&done) {
        assert_eq!(Some(a), b.as_ref());
    }
}

/// Avec `market=late`, un nœud de l'arbre sert à tous les mondes qui l'atteignent, quelle que soit
/// la carte arrivée au marché : ses coups doivent y être possibles (sinon la descente échoue). Des
/// recherches assez longues pour repasser souvent sous les choix de peuple, à toutes les tailles.
#[test]
fn market_late_trees_stay_legal_in_every_world() {
    let searcher = az("n=150,market=late,crn=1");
    for n in 2..=5 {
        let table = Table::new(n, 600 + n as u64);
        for (g, k) in positions(&table, 13, 5, 40) {
            searcher.search(&g, &k, &mut Rng::new(n as u64), true, &mut Evals::default());
        }
    }
}

/// Des parties entières où chaque place cherche avec toutes les options : les fins simulées vont au
/// bout, et l'histoire vide (`history=0`) ne gêne pas la recherche.
#[test]
fn whole_games_with_every_option() {
    let agent = az("n=4,roll=1,market=late,crn=1,win=0.75,mix=0.5,explore=1,history=0");
    for n in [2, 3] {
        let table = Table::new(n, 500 + n as u64);
        let seats: Vec<&dyn Agent> = (0..n).map(|_| &agent as &dyn Agent).collect();
        let played = play_game(&table, 3, &seats, 11);
        assert_eq!(played.outcome.players, n);
        assert!(played.moves > 10 * n, "{} décisions", played.moves);
    }
}

/// Sans histoire (`history=0`), le joueur de l'arène voit ce que voit l'Expert du jeu : ce que
/// donne `Knowledge::of` à chaque décision, quoi que l'arène lui ait appris.
#[test]
fn without_history_the_agent_sees_what_the_game_expert_sees() {
    let blind = az("n=32,history=0");
    let told = az("n=32");
    let table = Table::new(4, 77);
    let mut differ = 0;
    for (g, k) in positions(&table, 21, 31, 12) {
        let a = blind.act(&g, &k, &mut Rng::new(8));
        let b = told.act(&g, &Knowledge::of(&g.state), &mut Rng::new(8));
        assert_eq!(a, b);
        differ += usize::from(told.act(&g, &k, &mut Rng::new(8)) != a);
    }
    // l'histoire change bien des décisions (sinon ce test ne vérifierait rien)
    assert!(differ > 0, "aucune décision ne dépend de l'histoire");
}

/// Le selfplay sans bruit après la première manche (`noisy_rounds`), fins simulées dans les
/// recherches complètes seulement : la partie va au bout et enregistre ses positions ; les coups
/// des recherches complètes y sont leurs coups sans bruit.
#[test]
fn selfplay_without_noise_after_the_first_round() {
    use smallmobile_lab8::az::Direct;
    use smallmobile_lab8::selfplay::{self, Options};
    let teacher = az("n=12,roll=1");
    let opts = Options { noisy_rounds: 1, ..selfplay::RECIPE };
    let table = Table::new(3, 700);
    let mut sink = std::io::sink();
    let played =
        smallmobile_lab8::az::now(selfplay::play(&table, 4, &teacher, opts, &mut Rng::new(9), &mut sink, &Direct))
            .expect("partie");
    assert_eq!(played.outcome.players, 3);
    assert!(played.positions > 5, "{} positions", played.positions);
}

/// La part du hasard (`--luck`) se calcule à côté de la partie, sur son propre générateur : les
/// mêmes coups avec ou sans elle. Chaque correction est de somme nulle sur les joueurs (des parts de
/// victoire), et des coups à hasard (choix de peuple, dés) en donnent.
#[test]
fn luck_leaves_the_games_unchanged_and_sums_to_zero() {
    use smallmobile_lab8::az::Direct;
    use smallmobile_lab8::selfplay::{self, Options};
    let player = az("n=8");
    let table = Table::new(4, 710);
    let run = |luck: bool| {
        let opts = Options { luck, ..selfplay::RECIPE };
        smallmobile_lab8::az::now(selfplay::play(
            &table,
            6,
            &player,
            opts,
            &mut Rng::new(2),
            &mut std::io::sink(),
            &Direct,
        ))
        .expect("partie")
    };
    let (plain, lucky) = (run(false), run(true));
    assert_eq!(plain.moves().collect::<Vec<_>>(), lucky.moves().collect::<Vec<_>>());
    assert_eq!(plain.positions, lucky.positions);
    assert!(plain.compact.luck().iter().all(|l| l.iter().all(|&x| x == 0.0)));
    let luck = lucky.compact.luck();
    for l in luck {
        assert!(l.iter().sum::<f32>().abs() < 1e-4, "{l:?}");
    }
    assert!(luck.iter().filter(|l| l.iter().any(|&x| x != 0.0)).count() > luck.len() / 2, "{luck:?}");
}
