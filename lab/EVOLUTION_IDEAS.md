# Expert AI: directions for further development

Research proposals from the review of 20 October 2026. Measured results and campaign status
belong in [JOURNAL.md](JOURNAL.md); this document explains the proposed next steps, their
motivation, and how to test them. The proposals below have not been validated in SmallMobile.

The objective is stronger play on generated maps with 2–5 players, both with a generous
laboratory search budget and within the game's latency budget.

## Assessment of the current direction

The recent change in training data has produced a meaningful improvement. Earlier
generations were yielding diminishing returns. Learning from stronger games and renewing
the value targets broke that pattern:

- The P2 value branch, C (`p2c`), improved over gen8. The gain survived a second training seed
  and evaluation on fresh deals.
- Gen9 improved over C, with a score of +7.9 ± 1.7 points across 920 deals. Adding the new
  policy targets also helped compared with updating the value targets alone.
- Gen9 scored +11.1 ± 3.8 against gen8, but that arena stopped early after 184 deals.
  Treat this as an early estimate of the gain's magnitude.

These scores are percentage points above the equal-strength expectation of 1/n wins,
averaged equally across the four player counts. The intervals are approximate 95%
intervals computed with deals as the independent units. Gains against different opponent
fields cannot be added together.

The most informative ablation concerns the source of value supervision. In the P2 pilot,
actual strong-game outcomes helped more than averaged policy-only continuations; combining
both sources worked best. A continuation played by the network without search is not as
strong as an Expert game. Averaging 32 such continuations reduces randomness, but does not
remove their strategic bias. Preserve actual game outcomes as a central training signal.
These ablations change training sources, not isolated frozen heads: policy and value still
share a trunk.

The gen10 comparison between full history and recent strong-game data is a useful test of
the replay recipe. No gen10 strength result was available at the time of this review.
Do not infer that older positions are useless merely because their old labels are less
useful: retaining selected positions and refreshing their targets is a separate possibility.

The evidence supports continued training with the new recipe. It does not yet establish
robustness against diverse human strategies. Recent arenas mainly compare related agents
using the shipped search settings; the few human and LLM games are diagnostic examples,
not a reliable strength benchmark.

## Immediate opportunity: bring opponent modelling into the game

The shipped configuration still uses `rp=2`: the network plays its own continuations,
while Normal plays every opponent's continuations. The laboratory's `rp=4` uses the known
type of each seat instead.

With gen8, `rp=4` at 80 iterations beat the shipped 200-iteration configuration at comparable
mean compute time among Experts, and also performed well at mixed tables. This is the most
direct opportunity to turn an existing laboratory finding into stronger production play.

First repeat the comparison with gen9, including strong homogeneous fields and mixed tables.
Then calibrate latency on a small sample of production-representative decisions on mini.
Human seats need separate evaluation: the current fallback is a fixed mixture, not a
validated human model. Do not assume the gen8 search gain adds to the gen9 training gain.

Relevant code: [Expert configuration](../brain/src/expert.rs),
[opponent models](../brain/src/opponents.rs), and
[server integration](../src/server/expert.ts).

## 1. Train against a population, including agents that expose weaknesses

**Hypothesis:** a population of opponents will produce a more robust strategy than training
mainly against the current network.

Self-play can leave shared blind spots: both sides may rarely attempt a strategy that
exploits the other. The Diplomats game is a useful example of a situation to investigate,
although a single game does not establish a general weakness.

Keep a small population of frozen generations and different playing behaviours. Later, add
specialist opponents whose objective is to beat the current champion, even if they are weak
against the rest of the population. Such opponents can reveal strategies that broad
average-strength selection would discard.

**First experiment:** compare the current self-play recipe with a fixed mixture of opponents
at the same total compute budget. Keep network architecture, targets, and training updates
comparable. Evaluate against a held-out panel of strong agents and mixed tables, including
opponents absent from training. Report performance by opponent and player count, not only
the average.

**Success criterion:** improvement across the held-out panel without a substantial collapse
against a particular style. Winning against the immediately preceding generation alone is
insufficient. Multiplayer matchups can depend on table composition; a single scalar rating
may hide useful information.

A second, separate experiment would condition policy and value on known or inferred opponent
profiles. The same board can have different prospects against different responses. Today,
opponent models can affect rollouts, but their identities or beliefs are not network inputs.
Use public seat types and observed behaviour, with uncertainty for unknown players.
Do not attempt to learn a general human model from the handful of archived human games.

The inspiration is AlphaStar's population and exploiter training. Its success is evidence
for the approach in another game, not a guarantee for SmallMobile:
[the authors' explanation](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/).

## 2. Teach the network what produces the final result

**Hypothesis:** learning regional and intermediate outcomes will make each game more useful
than supervising only the action policy, winner, and future coins.

The current network already predicts future coins, so merely adding a score head would not
be new. The proposed change is to decompose that prediction into more local, informative
targets, such as:

- Who owns each region after a complete rotation of players.
- Whether an army retains its territories and how many tokens it loses.
- Income at the next scoring event, and future income contributions by region.

A final defeat gives little information about which region was misjudged. These targets
would provide more local feedback about defence, expansion, and decline timing. Keep winning
as the main objective; use the additional predictions to improve the shared representation.
Some auxiliary heads could be used only during training, avoiding their cost during play.

**First experiment:** derive a small set of targets from recorded strong games, then compare
training with and without them using the same games, updates, and main losses. Start with
ownership after a full rotation and next-scoring income. Measure value calibration and
regional prediction errors, but retain arena strength at comparable runtime as the decision
criterion.

KataGo's ownership and score targets motivate this proposal. SmallMobile accumulates income
over time, so final-board ownership alone is unlikely to be the right translation:
[KataGo, section 4.1](https://arxiv.org/html/1902.10565v5#S4.SS1).

### First audit whether the inputs contain the necessary public information

Code inspection identified concrete omissions in [encode.rs](../brain/src/encode.rs):

- Peace relationships are encoded only when they involve the player whose perspective is
  being encoded. Peace between two opponents is absent.
- The network receives the number of known cards at the bottom of the race deck, but not
  the public order of those returned cards.
- Opponent type or behavioural beliefs are not supplied to the network.

The simulator can use information that the network does not receive. In particular, two
states that differ in an opponent-to-opponent peace agreement can otherwise look identical
to the network while having different strategic consequences.

First construct controlled examples to verify these representation collisions, then measure
their frequency and impact. Test encoding corrections separately from auxiliary targets.
Residual value error alone does not prove that missing information is its main cause.
Only encode public knowledge; hidden deck order and future dice remain sampled.

Relevant code: [encoder](../brain/src/encode.rs), [network](py/model.py),
[training losses](py/train.py), and [game recordings](src/selfplay.rs).

## 3. Let the agent generate and evaluate its own placement plans

**Hypothesis:** the current action catalogue excludes strategically valuable plans.

In [moves.rs](../brain/src/moves.rs), heroes, fortresses, and encampments are placed by a
fixed heuristic. Redeployment is selected from a small catalogue, and defence has at most
two candidate distributions. Search cannot discover a plan that is never proposed,
regardless of the number of training generations.

The longer-term direction is a plan generator that proposes several legal allocations,
followed by search that evaluates them. Score the actual content of a plan—token allocations,
protected regions, diplomatic choices—rather than relying only on a fixed plan index.
A sequential or hierarchical generator could explore this space without enumerating every
possible allocation.

**First experiment:** build a set of positions where defence and special placements matter.
Add local alternatives to the existing candidates: move a token, protect a different region,
or redistribute camps. Compare their consequences with a stronger evaluator, then test the
expanded catalogue in an arena. Keep the existing plans as candidates.

**Success criterion:** repeated improvements on held-out positions and a measurable arena
gain at comparable time. The restriction is certain; its practical cost is not yet measured.
If simple additional candidates rarely help, a learned generator should move down the list.

Account for action-key semantics, training format compatibility, and branching cost before
extending this experiment into an architecture change.

## 4. Branch on public observations inside search

**Hypothesis:** distinguishing observed outcomes will improve decisions whose future options
depend strongly on dice or market draws.

The current search is an open-loop tree of action keys. An edge has one child even when
different sampled worlds produce different public observations after that action.
For example, the same market-slot key can refer to different race/power combinations.
This shares statistics efficiently, but can mix situations that should lead to different
continuations.

Investigate an observation-conditioned tree: after a random event becomes visible in a
simulation, allow subsequent decisions to depend on that observation. This must not expose
the actual hidden deck order or future dice. Branch only on information that would be public
at that point in the simulated game.

**First experiment:** keep the network fixed and use a small collection of controlled
positions with consequential market or dice outcomes. Compare the current tree with
observation splitting at those events. Measure decision quality, effective visits per
branch, memory, and runtime before attempting a full rewrite.

**Risk:** extra branches may dilute an already small search budget. A more faithful tree is
not automatically stronger at 200 iterations. Promote the approach only if its benefit
survives a comparison at equal time.

Relevant code: [search tree and traversal](../brain/src/search.rs),
[action keys](../brain/src/moves.rs), and [public knowledge](../brain/src/know.rs).

## Suggested order and evaluation discipline

1. Finish the gen10 experiment and interpret the replay comparison on its own terms.
2. Validate gen9 with `rp=4` and measure the production latency tradeoff.
3. Run a small population-training experiment as the next change to the learning recipe.
4. Audit public-state encoding, then test territorial auxiliary targets separately.
5. Probe the placement catalogue with additional candidates before building a learned planner.
6. Test observation-conditioned search on controlled cases before a general rewrite.

For each experiment, fix the hypothesis, comparison, seeds, opponents, and budget in advance.
Use independent confirmation deals, rotate seats, retain per-player-count results, and
compute uncertainty by deal. Compare both strength and runtime. A lower training loss or a
better score against Difficile alone is not enough to justify promotion.

Keep strong-game outcomes in the training mix. Treat population diversity and better targets
as experiments, not reasons to abandon a recipe that has just demonstrated progress.

Lower priorities for now are a much larger network, more averaging of the same policy-only
continuations, and many unchanged generations without checking their marginal gain.
The earlier larger-network failure used older targets; revisit capacity if the new data
or diagnostics give a reason to do so.

Run heavy experiments on AWS through `lab/tools/cloud.sh`, following
[AGENTS.md](../AGENTS.md) and the budget and experiment protocol in
[JOURNAL.md](JOURNAL.md). Keep manifests, detailed results, and decisions with each campaign.
