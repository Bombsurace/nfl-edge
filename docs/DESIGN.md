# Design notes

## Why Elo instead of a from-scratch classifier

With ~800-900 training games and no play-by-play/injury features wired
up, a walk-forward Elo system (following FiveThirtyEight's NFL Elo
methodology closely: margin-of-victory multiplier, home-field advantage,
1/3 between-season regression to the mean) is a harder baseline to beat
than it looks, and it comes with three properties this project needed:

1. **No lookahead by construction.** Every game's prediction is computed
   from ratings *before* that game updates them. A backtest of it is
   honest by default -- there's no way to accidentally leak a game's own
   result into its own prediction, which is the single easiest mistake
   to make in this kind of project.
2. **Built-in recency weighting.** Each result nudges a team's rating;
   older results get progressively overwritten. That's a direct, testable
   answer to "should recent seasons count more" without a bolted-on
   weighting scheme.
3. **Fully inspectable.** `model.elo.ratings_table()` prints one number
   per team you can sanity-check against your own sense of the league.

## The season-regression bug this build caught

Early on, generating a Week 1 2026 card produced absurd EVs (60-80% on
several games). The cause: `EloModel.predict()` was being called for a
new season using end-of-2025 ratings with no between-season mean
reversion applied, because that reversion only happened inside
`fit()`'s loop over *completed* games -- and 2026 had none yet. A team
that had a historically bad 2025 (say, a rating in the 1270s) was being
treated as if it would obviously stay that bad, instead of regressing
1/3 of the way back toward the 1505 baseline the way every other season
transition does. Fixed via `EloModel.advance_to_season()`, called
explicitly before predicting into a season with no completed games. This
is exactly the kind of bug walk-forward backtesting is designed to
surface -- it's now covered by the backtest path (which processes every
season transition through the same `fit()` loop) and by the Week 1
smoke test in the repo's test notes.

## Why Platt scaling instead of isotonic regression for calibration

The brief's section 7 mentions `p_home_cal` without specifying a method.
`sklearn.isotonic.IsotonicRegression` was the obvious first choice, but
with ~800 training games it produced wide flat "plateaus" -- multiple
genuinely different games (different teams, different weeks) mapped to
the *exact same* calibrated probability, because isotonic regression has
no smoothness prior and will carve out a flat region wherever the noisy
empirical win rate happens to be locally non-monotonic. That's the
calibrator overfitting noise in bucket boundaries, not a real finding.
Platt scaling (a 2-parameter logistic fit on the logit of the raw
probability) can only stretch or compress a sigmoid, so it can't produce
that artifact. It's now the default (`calibration_method="platt"`);
isotonic is still available (`calibration_method="isotonic"`) for
comparison once there's enough data (multiple more seasons) for it to be
trustworthy -- this is exactly the kind of thing to re-test periodically
as the training window grows.

## The headline backtest finding, and what it implies

See the README's "What the 2023-2025 backtest actually found" section
for the numbers. The short version: the "bet whichever side has higher
computed EV" rule from the brief disproportionately selects underdogs
(69% of picks), and those underdog picks lose money (-12.3% ROI) while
favorite picks are close to breakeven (+0.8% ROI). Tightening the
underdog edge requirement doesn't fix it -- it's not a threshold
problem, it's that the model's probability estimates are least reliable
exactly where they disagree with the market the most.

### Ideas worth backtesting next (in rough priority order)

1. **Shrink toward the market.** Blend the model's probability with the
   market's no-vig implied probability before computing EV (e.g.
   `p_final = 0.3 * p_model + 0.7 * p_market_novig`). This is standard
   practice in sports modeling specifically because it encodes "the
   market usually knows more than I do, especially on the tail," and it
   directly targets the failure mode found above. Would need a small
   `p_market_novig` helper in `odds.py` (remove the vig from both sides'
   implied probabilities so they sum to 1).
2. **A separate, stricter model for underdog bets**, or simply excluding
   underdogs above some odds threshold from consideration entirely,
   rather than trying to fix them with an edge-requirement dial.
3. **Situational features (brief section 14/15)** -- rest, short week,
   travel, divisional games. Don't add these because they sound useful;
   the brief is right that sportsbooks already price most of this in.
   Test each one by adding it as an Elo/calibration input and comparing
   the walk-forward backtest with vs. without, on the same games.
4. **Training window sensitivity.** Compare `TrainingWindow(seasons=...)`
   choices (e.g. 2022-2025 vs. 2023-2025 vs. season-weighted) via the
   same backtest, holding everything else fixed. The warmup period
   (`warmup_seasons`, default 8) is separate from this and controls only
   how much history the Elo ratings themselves see before the evaluation
   window starts -- it likely doesn't need to change.
5. **Closing-line value.** `db.first_and_last_snapshots()` is already
   wired up to compare a card's first pull to its last before kickoff --
   once a season of live snapshots accumulates, check whether the
   model's picks tend to move *with* or *against* the closing line
   (brief section 15/16). If the line consistently moves toward the
   model's side after the model's pick, that's a much stronger signal
   than backtested ROI alone.

Every one of these should be evaluated with `walk_forward_backtest()` on
the full training window before being adopted -- not on the strength of
one good week, per the brief's own section 19.

## The multi-factor model: QB, weather, home-field

Built in response to wanting adjustable emphasis per factor (QB trust,
weather, home/away) rather than a single opaque Elo number. `factors.py`
adds each as its own standardized feature combined through a logistic
regression, with a slider-style multiplier on top of whatever coefficient
the data actually fit -- see the README's "interactive tuner" section for
the page this powers.

### A mechanical bug the QB feature construction had to avoid

The first version of the QB factor computed, for each game, "the
starting QB's own rating minus their team's current Elo rating" -- the
idea being that a backup starting for an injured QB1 would show up as a
negative gap. Alone, this feature *predicted home_win with the wrong
sign* (-1.22 logistic coefficient on its own, before even touching
Elo). The cause: team Elo updates with K=20, the QB rating with a
smaller K=8, so team Elo simply moves faster than QB rating in either
direction. During any hot streak, team Elo runs ahead of the QB's own
(slower) rating, making the "gap" go negative for reasons that have
nothing to do with who's playing quarterback -- a pure artifact of the
two ratings updating at different speeds, not a real finding.

Fixed by switching to a direct QB-vs-QB rating differential (home QB
rating minus away QB rating, no team-relative subtraction), which is
positively signed on its own (+0.76), matching intuition. This is
exactly the kind of check that's easy to skip and easy to get wrong --
"the model output moved in a plausible-looking way" is not the same as
"the feature is measuring what it claims to."

### Two more factors: rest advantage and divisional games

Added alongside a bug fix described next: `rest_diff` (home rest days
minus away rest days, /7 -- a short week off a Thursday game shows up
strongly negative for the team facing it, a bye-week return shows up
positive) and `div_game` (the source data's existing 0/1 divisional-
matchup flag, fed in as-is to let the fit decide whether familiar
rivals really do play closer games, rather than assuming it). Both were
cheap to add -- the raw data already has full coverage for
`home_rest`/`away_rest` and `div_game`, so no new data source was
needed, unlike weather forecasts for future weeks (see the README's
odds-sources section for that gap).

### A real bug found while adding those two: weather and home-field sliders did nothing

While wiring up the two new factors, a JSON-key mismatch turned up
between `export_client_data.py` (which had been exporting weather under
the key `"weather"`) and the tuner page's JS (which reads
`game['weather_severity']` -- a different key, so it silently evaluated
to 0 for every game, backtest and upcoming week alike). Separately,
`home_extra` was never included in the per-game JSON at all -- omitting
a "constant 1.0" field looked safe when it was first written, but the
JS reads `game[f.key]`, which is `undefined` (i.e. 0) when the key is
simply missing. Net effect: **the weather and home-field sliders had
zero effect on any number on the published page, in every version
published before this fix** -- not "small effect," not "only matters
for future games," literally multiplied by zero every time, on both
tabs. Confirmed fixed with a before/after check: dragging the weather
slider to 5x on the full backtest tab now moves "all picks" ROI (was
frozen regardless of the slider; now moves from -7.4% to -6.5% as the
slider changes). Both keys are correct now, and this class of bug -- a
slider that silently does nothing because of a naming mismatch between
Python and JS -- is worth specifically re-checking any time a new
factor is added to `FACTOR_COLS`.

### Does it actually help? (bootstrap + walk-forward, both honest)

Fitting all six factors together (in-sample, full 2023-2025 window,
854 games) and bootstrapping 90% confidence intervals on each
coefficient (300 resamples):

| factor | coefficient | 90% CI | reads as |
|---|---|---|---|
| elo_diff | +2.00 | [+1.66, +2.47] | confidently positive |
| qb_diff | -0.13 | [-0.55, +0.28] | crosses zero -- inconclusive |
| weather_severity | +0.10 | [-0.60, +0.74] | crosses zero -- inconclusive |
| home_extra | -0.03 | [-0.10, +0.04] | crosses zero -- Elo's built-in HFA already covers it |
| rest_diff | +0.24 | [-0.13, +0.58] | crosses zero -- inconclusive, but leans positive |
| div_game | -0.06 | [-0.30, +0.19] | crosses zero -- inconclusive |

Only team strength clears the bar. That's not a failure of the
project -- it's the honest answer at this sample size, and it's exactly
why the interactive page reports these intervals next to each slider
instead of presenting 1x as settled fact. `rest_diff` is the one worth
watching as more seasons accumulate -- its interval already leans
clearly positive even though it still crosses zero.

A walk-forward sweep (`factors.walk_forward_multifactor_backtest`, base
weights refit weekly on prior games only) across a few multiplier
settings, full 855-game window (six-factor version -- the "default" row
differs slightly from the four-factor number reported before rest/div
were added, since rest and div are now part of every configuration at
their own 1x):

| configuration | win% | ROI |
|---|---|---|
| default (all 1x) | 39.5% | -8.32% |
| default, flagged only | 36.3% | -7.44% |
| QB off (0x) | 37.5% | -8.63% |
| QB 2x | 40.9% | -5.33% |
| weather off (0x) | 38.1% | -8.71% |
| weather 2x | 39.8% | -7.41% |
| QB + weather both off | 36.8% | -9.35% |

Turning QB emphasis up modestly reduced losses in this particular sweep
(-5.33% vs. -7.67% default) -- interesting, but with a bootstrap
interval that crosses zero, this could easily be this particular 855-
game sample rather than a real, stable effect. Worth continued tracking
as more seasons accumulate, not worth concluding on. **The important
line in that table is that every single configuration is still net
negative** -- none of these secondary factors fix the core underdog-
overconfidence problem from the base model; at best they shave a couple
points off the loss. Don't let a slider showing a smaller loss read as
"found the edge."

### Next steps for this part specifically

- More seasons will narrow these confidence intervals -- re-run the
  bootstrap and the walk-forward sweep periodically as 2026 completes,
  rather than assuming the current inconclusive verdict is final.
- The QB feature is a proxy (see qb.py's docstring) -- a real
  player-value model (EPA/CPOE-based, the way 538/nfelo build theirs)
  would need play-by-play data this project doesn't pull in yet. Worth
  it only if the current proxy's coefficient starts looking stable and
  meaningfully different from zero.
- The weather feature currently doesn't distinguish "home team's usual
  climate" from "conditions today" -- a Buffalo home game in the snow
  and a Miami home game in a freak cold snap get the same severity
  score today, but arguably shouldn't. Testable extension, not yet
  built.
- Weather forecasting for upcoming weeks and spread betting are both
  now built -- see the two sections below for how each works and,
  importantly, weather's real operating limitation in this project's
  own hosting environment.

## Live weather forecasts for upcoming games

nflverse's `temp`/`wind` columns only ever record the *actual,
post-game* conditions -- always NaN for a game that hasn't been played
yet, which is exactly when the tuner's "this week" tab needs weather.
`weather_forecast.py` fills that gap with a live call to Open-Meteo
(free, no API key), keyed by each game's stadium coordinates
(`stadiums.py`, keyed by nflverse's `stadium_id` rather than team, since
a few games are played at neutral international sites) and its local
kickoff date/time (`gameday`/`gametime`, matched to Open-Meteo's
`timezone=auto` hourly forecast so no manual timezone math is needed).
Domes and closed roofs skip the forecast call entirely (same as the
historical path -- severity is 0 regardless of outside conditions).

**This needs outbound network access to `api.open-meteo.com`, and that
access is not guaranteed everywhere this project runs.** Concretely: the
cloud sandbox this project's own weekly scheduled refresh runs in
restricts outbound requests to an allowlist (package registries and a
short list of specific hosts) that does not include general third-party
APIs -- a call to Open-Meteo from there fails immediately with a
`ProxyError`, every single time, not intermittently. `fetch_forecast`
treats that identically to any other failure (network down, game more
than ~16 days out and past Open-Meteo's forecast horizon, unknown
stadium): log it, return `(None, None)`, and let the caller fall back to
the same neutral (severity 0) treatment an indoor game gets -- **never**
a fabricated number standing in as if it were real. The payload reports
exactly how many games got a real forecast each run
(`next_week.n_forecast_used` / `n_forecast_attempted`), and the tuner
page surfaces that count directly under the picks table, so "weather
was 0 for every game" is never silently indistinguishable from "every
game happened to be calm." Run this from a normal machine (a laptop, a
CI runner, any host without that outbound restriction) and it works as
designed -- verified offline against a mocked Open-Meteo response
(exact-hour match and nearest-hour rounding both correct) since a live
call isn't possible from this project's own dev/build environment
either.

## Spread betting

Same shape as the moneyline model -- predict something, compare it to
the real market, compute EV and guardrails -- for a different market.
`spread.py` fits a **linear regression** (not logistic) of home margin
of victory (`home_score - away_score`) on the same six factors, instead
of a win-probability logistic fit. Turning a predicted margin into a
cover probability needs one more assumption: actual margin is treated
as approximately normal around the predicted margin, with a standard
deviation fit from that same data's residuals (falls back to ~13.5
points, a commonly cited approximation of full-game NFL margin spread,
if there isn't enough data yet). That's a real simplification -- NFL
margins have a bump right around 3 and 7 points from how often games
are decided by a field goal or a touchdown-plus-extra-point -- but it's
the standard, disclosed way to get from "a predicted margin" to "a
probability of covering a specific line," in the same spirit as
`weather_severity`'s 0-1 approximation elsewhere in this project.

**Sign convention, verified rather than assumed:** in this dataset,
`spread_line` is the home team's *expected margin* -- positive means
the home team is favored by that many points. Checked two ways before
trusting it: regressing actual margin on `spread_line` gives a positive
slope (+0.57, moving the same direction, as an efficient market's line
should), and home teams already known to be big moneyline favorites
have large positive `spread_line` values. Home covers when
`actual_margin > spread_line`; a push (landing exactly on the line) is
possible on whole-number lines and isn't separately modeled, since most
real lines here carry a half point specifically to prevent it.

Once there's a cover probability and the market's real spread odds
(`home_spread_odds`/`away_spread_odds` -- genuine per-game numbers from
the same free data source as everything else, typically near -110 but
not assumed to be), the exact same `ev.evaluate_side` guardrail/EV
machinery the moneyline path already uses handles the rest -- no
separate spread-specific guardrail code was needed. One honest
modeling wrinkle worth flagging: `home_extra` (the constant-1.0 "extra
home-field emphasis" feature) is **mathematically unidentifiable** in
this linear regression -- a constant column is perfectly collinear with
the intercept once the data is centered (which `LinearRegression`'s
default `fit_intercept=True` does), so its coefficient is forced to
exactly 0, not just small. That's not a bug to fix; it's a real
property of OLS with a constant regressor. The moneyline logistic fit
doesn't have this problem (L2-regularized, so it still gets a small
non-zero shrinkage estimate) -- the tuner's "extra home-field emphasis"
slider genuinely has zero effect specifically on spread picks as a
result, and the page says so rather than silently rendering an inert
slider as if it did something.

**What the walk-forward spread backtest actually found** (six factors,
all at 1x, margin model refit weekly on prior games only, 855 games):
47.4% against-the-spread win rate, -9.2% ROI. The predicted margin
correlates strongly with the market's own spread line (r=0.76, a sanity
check that the sign convention and the model's general behavior are
correct) -- but the same disagree-with-the-market-and-be-wrong-more-
often-than-right pattern documented for moneyline underdogs above shows
up again here. This is a second, independent piece of evidence for the
same underlying finding: this model's edge, where it disagrees most
with an efficient market, is disproportionately the model being wrong.
Confirm any particular multiplier setting for real with
`python -m nfl_edge.cli spread-backtest --qb <mult> --weather <mult> --rest <mult> --div <mult>`.

## Offensive and defensive EPA/play

Every factor up to this point summarizes a team by its *results* (Elo)
or its roster (QB rating). None of them look at how the team is actually
playing snap to snap. `epa.py` adds that: **expected points added (EPA)
per play**, split into an offensive number and a defensive-EPA-allowed
number, computed from real play-by-play data (nflverse's public pbp
release, free, no key -- the same project this data source already
belongs to, just a different, much larger file than `games.csv`). EPA
per play is the standard efficiency stat across public NFL analytics
(nflfastR, rbsdm.com, and similar) -- it credits a play by how much it
changed the offense's expected points, so a 4-yard gain on 3rd-and-3 (a
first down) counts very differently from a 4-yard gain on 3rd-and-8 (a
punt coming), which raw yardage can't distinguish.

Two factors come out of this, both signed positive-means-home-edge like
every other factor here: `epa_off_diff` (home's rolling offensive
EPA/play minus away's) and `epa_def_diff` (away's rolling defensive
EPA/play *allowed* minus home's -- higher allowed EPA is a worse
defense, so this being positive is a home edge). "Rolling" means
season-to-date, computed the same walk-forward way as Elo and QB rating
-- every game's feature value uses only that team's *prior* games, never
the game itself or anything later, so there is no lookahead here either.
Two judgment calls, not fit from data, keep it well-behaved: a new
season starts from 40% of a team's carried-over rating rather than a
hard reset (`SEASON_CARRYOVER`, since a roster changes but doesn't
become a different team overnight), and any team's rolling number is
shrunk toward the league-average prior of 0.0 by an amount equivalent to
4 "phantom" games (`SHRINKAGE_GAMES`), so week 1 or 2 of a season isn't
whiplashed by one fluky result.

**Stated plainly, the biggest limitation:** this is *not* opponent-
adjusted. A team's offensive EPA reflects who it has actually played,
not a neutral schedule -- three games against bad defenses inflates it,
three games against good ones deflates it, and nothing here corrects for
that yet (a proper fix looks like iterative opponent adjustment -- SRS,
or a small Elo-style rating per side -- future work, not built). It also
doesn't filter garbage time, so a blowout's prevent-defense snaps count
the same as a tied game's fourth-quarter snaps. Same posture as QB
rating and weather elsewhere in this project: a real, disclosed
approximation, not a finished stat, and it earns its coefficient in the
backtest rather than being assumed to matter.

**What actually happened when it was added.** In the moneyline model,
`epa_off_diff` is the *second* factor (after `elo_diff` itself) whose 90%
confidence interval clears zero in the current data (fit on 854 games):
roughly 0.09 to 1.29, a real, if modest, signal -- offensive efficiency
predicts wins here even after Elo already gets a vote. `epa_def_diff`'s
interval still crosses zero; defense-allowed EPA isn't yet distinguishable
from noise at this sample size. In the spread model, both EPA
coefficients have wide intervals that cross zero in the single-fit-on-
whole-window view the tuner page uses (plausibly some multicollinearity
with `elo_diff` -- good teams tend to have both good Elo and good EPA,
which inflates the uncertainty on each individually even when their
combined contribution is real) -- but the walk-forward backtest, which is
the honest number, still moved in the right direction on both models:
moneyline all-picks ROI improved from -8.3% to -6.9% (win rate 37.9% ->
40.2%), and spread all-picks ROI improved from -9.2% to -6.4% (ATS win
rate 47.4% -> 48.9%), all six factors at 1x, EPA added as the seventh and
eighth. Real, worth keeping -- and still nowhere near enough to make this
project profitable on its own. Confirm any setting for real with
`python -m nfl_edge.cli multifactor-backtest --epa-off <mult> --epa-def <mult>`
or the `spread-backtest` equivalent.

## Pass-rush pressure, approximated by sack rate

Second on the priority list after EPA: does the trenches battle matter
beyond what EPA already picks up? `pressure.py` adds a sack-rate factor,
built from the same nflverse play-by-play release EPA uses -- sacks taken
per dropback (protection) and sacks recorded per opponent dropback (pass
rush), where "dropback" is nflverse's `qb_dropback` flag (pass attempt +
sack + scramble -- the standard denominator, since a sack ends the play
before an "attempt" is logged and using attempts alone would undercount).

**Naming honesty, stated up front:** this is *pressure approximated by
its most visible outcome*, not pressure itself. True pressure (a QB
hurried or hit but never sacked) isn't in any free public data source --
that requires proprietary charting data (PFF, NFL Next Gen Stats) this
project has no access to. Sack rate is a real signal on its own -- it's
what actually shows up on the scoreboard and feeds directly into EPA --
but it understates a pass rush that generates constant pressure without
finishing the sack, and it can be inflated by a QB who holds the ball too
long rather than pure O-line or pass-rush quality.

Two factors come out of this, both signed positive-means-home-edge:
`sack_allowed_diff` (away's sack rate taken by their offense minus
home's -- positive means home's O-line has protected its QB better than
away's has) and `sack_generated_diff` (home's sack rate recorded by their
defense minus away's -- positive means home's pass rush has gotten there
more often). Same walk-forward, no-lookahead construction as EPA, and the
same two judgment-call constants doing the same job: `SEASON_CARRYOVER`
(40% of a team's sample weight kept across a season boundary) and
`SHRINKAGE_DROPBACKS` (30 phantom dropbacks of "prior belief = league-
average sack rate," about one game's worth, blended in so an early-season
number isn't whiplashed by one fluky game). The league-average prior
itself (`LEAGUE_AVG_SACK_RATE = 6.5%`) is a rough constant, not fit from
this project's own data -- it only sets how fast an under-sampled team's
rate anchors to a plausible league level, not the final answer. Same two
disclosed limitations as EPA too: not opponent-adjusted (three games
against a bad O-line inflates a pass rush's apparent quality), and no
garbage-time filter.

**What actually happened when it was added.** `sack_allowed_diff` (the
protection side) is the one with a real signal here, and it's a
meaningfully *stronger* one than most of the other post-Elo factors: its
90% confidence interval clears zero in both models -- roughly 0.08 to 0.80
in the moneyline logistic, and about 12 to 94 points of predicted margin
in the spread regression (fit on 854/847 games), the widest, most
confidently-positive interval of any factor besides Elo itself and
offensive EPA. That tracks with intuition -- a sacked drive loses yardage,
a down, sometimes the ball, all of which shows up directly in the score.
`sack_generated_diff` (the pass-rush side) is *not* distinguishable from
noise yet at this sample size -- its interval crosses zero in both models.
Walk-forward backtest, the honest number: moneyline all-picks ROI moved
from -6.9% to -6.6% (win rate flat at 40.2%) -- a small, mixed effect,
consistent with only one of the two sack factors actually carrying
signal. Spread all-picks ROI moved from -6.4% to -5.7% (ATS win rate 48.9%
-> 49.2%) -- the larger, cleaner improvement, matching the strong CI on
the margin-regression side. Confirm any setting for real with
`python -m nfl_edge.cli multifactor-backtest --sack-allowed <mult> --sack-generated <mult>`
or the `spread-backtest` equivalent.

## Travel distance and time-zone shift

Third and last item on the priority list (closing-line movement, the
fourth, turned out to be infeasible -- nflverse's odds file carries only
the final closing line, no opening line or movement history to build
against). `travel.py` adds two factors that, unlike every rolling factor
above, need no historical fitting or walk-forward state at all: a team's
home city and a stadium's coordinates are both fixed facts, known before
kickoff for every game including future ones, so there's no lookahead
question here -- same category of "already known" input as `div_game`.

`travel_distance_diff` is the away team's great-circle distance from
their own home stadium to this game's stadium, minus the home team's own
distance (normally 0, since they're playing at home -- nonzero only for a
true neutral-site game, London/Mexico City/etc., where both teams travel
and the diff correctly nets out to whichever team travelled farther), in
thousands of miles. `travel_tz_diff` is the same idea for time zones: the
away team's body-clock shift (this stadium's UTC offset minus their
home's) minus the home team's own shift, in hours/3. Time zones are fixed
STANDARD-time offsets with no DST calendar -- the U.S. mainland's DST-
observing zones all shift together so this mostly cancels out, except
Arizona (doesn't observe DST), which is off by an hour from its Pacific-
or Mountain-zone opponents for roughly the first two months of each
season. A team's "home stadium" is whichever stadium they've hosted the
most games at in the available nflverse data (2015-2025) -- correct for
all 32 current teams as of this writing.

**What actually happened when it was added -- the strongest single
addition yet, and a genuinely counter-intuitive one.** Walk-forward:
moneyline all-picks ROI improved from -6.6% to -5.0% (win rate 40.2% ->
42.0%), the biggest jump of any factor this project has added. Spread
improved more modestly, -5.7% to -5.1% (ATS win rate 49.2% -> 49.6%).
Both factors clear zero somewhere: `travel_distance_diff`'s 90% CI is
-0.42 to -0.05 in the moneyline model (significant, `travel_tz_diff`
crosses zero there); `travel_tz_diff`'s CI is -3.52 to -0.33 in the
spread model (significant, `travel_distance_diff` crosses zero there) --
each factor pulls its weight in a different model.

The counter-intuitive part: both coefficients are NEGATIVE. Popular
sports-betting folklore (and this project's own working assumption
going in) says the away team travelling farther, or crossing more time
zones, should burden THEM and help the home team -- a positive
coefficient. The data here says the opposite: more away-team travel
distance or time-zone shift is associated with the home team being LESS
likely to cover or win, at this sample size. This project's posture on
`weather_severity` applies word-for-word here -- the sign was
deliberately not assumed going in, exactly so a surprising result like
this one gets reported rather than quietly "corrected" to match
expectation. A few honest reasons this could be real rather than noise:
West Coast teams (frequent long-distance, big-tz-shift travelers) have
been unusually strong in 2023-2025 specifically, media/scheduling bias
means cross-country games skew toward marquee, higher-talent matchups
already partly captured by other factors, and rigorous outside research
on NFL travel effects has generally found them weaker and murkier than
the folk narrative claims. It could also simply be a 3-season, 12-
covariate sample producing a fluke sign -- worth re-checking once more
seasons accumulate, not worth over-explaining now.

One more honest wrinkle, found while decomposing the two factors: on the
spread model specifically, `travel_distance_diff` ALONE (tz multiplier at
0) walk-forwards to -4.0% ROI / 50.2% ATS -- better than the two factors
combined (-5.1% / 49.6%). The two are correlated (a cross-country game
usually means both big distance AND a big time-zone shift), and having
both active in the same linear regression likely dilutes each one's
individually cleaner signal a bit. Left both in at their data-fit weights
by default anyway, matching how every other paired factor in this
project works (EPA off/def, sack allowed/generated) -- picking the
single best-backtesting sub-combination after the fact is exactly the
kind of after-the-fact fitting the project's own discipline section warns
against. The sliders exist so you can try `--travel-tz 0` yourself and
see it.

Confirm any setting for real with
`python -m nfl_edge.cli multifactor-backtest --travel-distance <mult> --travel-tz <mult>`
or the `spread-backtest` equivalent.

## The track record: a locked, no-lookahead prediction log

Everything above is a *backtest* -- re-fit and re-evaluated on historical
seasons on demand, under whatever slider weights you're currently
holding. It answers "how would this configuration have done," which is
a different question from "what did the page actually tell me, before
kickoff, this season." `tracker.py` answers the second one, because
that's the only fair way to judge whether the model is worth anything
going forward: a prediction that's recorded, or even quietly adjusted,
after the result is known isn't a prediction.

Two rules make it a fair record rather than a hindsight report. First,
a game's pick is written exactly once -- the first time it shows up as
an upcoming game -- and is never touched again by a later run, even
though later runs have seen more data and would compute a different
number if asked fresh. Re-running the export mid-week just fills in
newly-final scores for *other*, already-locked games; it can't revise a
pick that's already on the board. Second, every locked pick uses
default (1x on every factor) weights, deliberately independent of
wherever the sliders happen to be sitting elsewhere on the page --
because a track record needs one fixed configuration to mean anything
across sixteen-plus weeks, not a moving target that could be nudged
into looking better in hindsight. Both rules exist for the same reason:
so this file can't accidentally (or conveniently) become a highlight
reel.

The pick logic is intentionally the plainest possible reading of "who
does the model predict wins": moneyline pick is whichever side has
model win probability >= 50%, spread pick is whichever side has
predicted cover probability >= 50% (from `spread.predicted_margin` and
`spread.home_cover_prob`). That's a different, simpler rule than the
edge-maximizing "best side" logic the ROI backtests use elsewhere in
this project -- there, the question is "which side, if any, is worth
laying money on"; here, the question is just "which side did the model
think would happen," so there's no edge threshold or betting guardrail
involved. A moneyline tie is recorded as `ml_actual: "tie"` and excluded
from the win/loss tally (not a loss); a spread push is recorded the same
way (`spread_actual: "push"`, excluded, counted separately).

Storage is a flat JSON file, `data/processed/tracked_predictions.json`,
keyed by season-week and then by `game_id`. Unlike `client_data.json`
(fully regenerated every run, gitignored) this file holds information
that genuinely cannot be recreated after the fact -- what the model said
*before* kickoff -- so it's deliberately NOT gitignored and is meant to
be committed and kept, the same way you'd keep a paper betting log. It
starts empty and tracks forward from whenever this feature first ran;
it does not and cannot backfill 2023-2025, since that would mean
deciding after the fact what the model "would have said" using a
feature set that didn't exist yet at each past moment -- a different
and murkier exercise than this file is for. The existing "Full backtest"
tab already covers that historical window honestly, under any slider
setting.

Grading piggybacks on the same weekly refresh cadence the rest of the
page uses: by the time the Wednesday refresh runs, every game through
the previous Monday night is final in nflverse's data, so a game locked
as "next week" the previous Wednesday is graded the following
Wednesday without any extra scheduling logic -- `_grade_pending` just
scans every locked-but-ungraded game against the current `completed`
DataFrame on every run, regardless of which week is currently "next."

The "Track record" tab on the page presents this as a running record
(moneyline record and against-the-spread record as KPI tiles) with a
per-week pill bar so you can tab through completed and in-progress
weeks and see the win/loss/pending state of every pick. One thing this
does NOT do yet, flagged here on purpose rather than built speculatively:
break the record down by team, to see if the model is more or less
reliable for specific teams. That's a reasonable follow-up once there's
enough graded history to say anything meaningful about it, not before.

**Confidence ranking, for real confidence-pool and "pick 'em" games.**
Each locked week also carries a `confidence` (moneyline) and
`spread_confidence` (against the spread) rank per game, computed by
`_assign_confidence` in `tracker.py`: sort that week's games by the
model's own pick probability (`ml_pick_prob` / `spread_pick_prob`),
most lopsided call gets the week's game count, closest to a coin flip
gets 1. Moneyline and spread are ranked independently on purpose --
the model doesn't always agree on which game is "safest" by each
measure, and a real confidence pool is usually one or the other, not
both at once. This obeys the same lock-once discipline as everything
else here: ranking happens the moment `_lock_new_games` adds a week's
games (in practice, all of a week's games arrive in one batch, since
that's how the schedule data is pulled), gets marked
`confidence_locked` on the week, and is never recomputed after that --
a rank you've already copied onto a real pool sheet doesn't get
reshuffled by a later run that's seen more data. The rare edge case
(a game added to a week's slate after that week was already ranked)
just leaves that one game's rank as `null` rather than disturbing
everyone else's numbers. The tuner page's Track record table shows
both ranks per game, sorted most-confident-first by default so it
doubles as a ready-made cheat sheet, with the underlying probability
printed in small text under each rank for anyone who'd rather eyeball
the raw percentage than the 1-through-N slot.
