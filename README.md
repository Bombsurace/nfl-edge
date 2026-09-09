# nfl_edge

An NFL expected-value betting system (moneyline and spread), rebuilt as
a real Python project instead of a Google Colab notebook. See
`docs/DESIGN.md` for why each piece is built the way it is, and keep
reading for setup and the week-to-week workflow.

## Why not stay in Colab?

Colab is great for one-off exploration. It's a poor fit once you want a
*recurring, trustworthy* system, for a few concrete reasons:

- **No real database.** CSVs on Drive don't let you query "show me every
  bet with edge > 10% in November" -- a SQLite file does.
- **No walk-forward guarantees.** It's easy to accidentally let a
  notebook cell peek at data it shouldn't (calibrating on games it's
  about to grade). A module you can unit-test doesn't have that problem.
- **Nothing is version-controlled.** When a guardrail number changes,
  you want a git diff and a commit message, not a vague memory of "I
  think I changed the edge threshold at some point in October."
- **Manual reruns.** A script with a CLI can be scheduled (cron, GitHub
  Actions, a calendar reminder) instead of requiring you to open a
  browser tab and click through cells.

This project keeps everything Colab was good for -- you can still `pip
install -e .` and import these modules into a notebook if you want
interactive plotting -- while fixing the above.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No API key is required to get started. Historical data (including real
closing moneylines back to 1999, and any lines already posted for an
upcoming week) comes free from a public nflverse dataset. See
**Odds sources** below for when you'll want a paid key.

## Weekly workflow

```bash
# 1. Pull the freshest schedule/odds data
python -m nfl_edge.cli refresh-data

# 2. Find the next unplayed week
python -m nfl_edge.cli next-week --season 2026

# 3. Build (and save) that week's card
python -m nfl_edge.cli card --season 2026 --week 2

# 4. Re-run step 3 anytime during the week to refresh odds (injuries,
#    line movement, new books posting) -- every run is saved as a new
#    snapshot, so you can later compare your first card to the closing
#    line (closing-line value, section 15 of the original brief).

# 5. After games finish, grade the week
python -m nfl_edge.cli grade --season 2026 --week 2

# 6. See how you're doing so far this season
python -m nfl_edge.cli report --season 2026
```

Everything lands in `data/nfl_edge.sqlite` -- open it with any SQLite
browser, or `pandas.read_sql` from a notebook, if you want to explore by
hand.

## Backtesting

The single most important question from the original brief was: **does
this actually have an edge, or does one good week just look like it
does?** `walk_forward_backtest()` answers that honestly:

```bash
python -m nfl_edge.cli backtest --out data/processed/backtest_2023_2025.csv
```

This predicts every game in the training window using only information
that would have been available before kickoff -- Elo ratings update
game-by-game with no lookahead, and the calibration curve is refit each
week using only *prior* weeks' games (the first ~100 games of the window
run uncalibrated, since there isn't enough prior data yet to fit a
calibrator on). It grades every pick with a flat $10 stake and reports
the five baselines from the brief (always home, always favorite, highest
model probability, highest EV, highest EV + guardrails) side by side.

### What the 2023-2025 backtest actually found

855 games, walk-forward, real closing lines. Headline:

| segment | n | win% | units | ROI |
|---|---|---|---|---|
| all model picks | 855 | 40.5% | -70.1 | -8.2% |
| guardrail-flagged only | 630 | 36.0% | -60.0 | -9.5% |

Both are solidly negative. Before you conclude "the whole idea is
broken," look at the split that explains it:

| | n | win% | ROI |
|---|---|---|---|
| favorites picked | 268 | 64.6% | **+0.8%** |
| underdogs picked | 587 | 29.5% | **-12.3%** |

**The "bet whichever side has the better EV" rule, as specified, picks
underdogs about 69% of the time -- and loses badly on them.** This is
the classic favorite-longshot bias trap: a market's underdog price is
already shaded to account for the public loving longshots, so whenever
a simple model disagrees with the market *most* on a big underdog,
that disagreement is far more often the model being wrong than the
market being wrong. The model is not equally reliable in both
directions, but the original "just take the higher-EV side" rule
implicitly assumes it is.

I also tested tightening the guardrail's underdog edge requirement a
lot (from the brief's +2% to +15% for any dog +100 or longer). It barely
helped -- flagged bets dropped from 630 to 425 but ROI stayed around
-10%. That's an important negative result: **this isn't a threshold
problem, it's a calibration problem.** The picks that still clear a
strict edge bar are exactly the ones where the model disagrees with the
market the most, and those are disproportionately the least reliable
ones. Raising the bar doesn't filter out the bad ones -- it can't,
because "high computed edge" and "reliable computed edge" are anti-
correlated here.

Favorites, meanwhile, are close to what you'd expect from an efficient
market with vig (roughly breakeven to slightly positive) -- which is
itself a useful sanity check that nothing is badly broken elsewhere in
the pipeline.

**Practical next step worth testing:** shrink the model's probability
toward the market's own no-vig implied probability (a weighted blend,
e.g. 70% market / 30% model) before computing EV, rather than trusting
the raw model probability outright. This is standard practice precisely
because it acknowledges the market usually knows more than a
score-history-only Elo model, especially on the long tail. That's the
natural next experiment -- see `docs/DESIGN.md` for other ideas (adding
QB/injury-aware features, testing a shorter training window, etc.), all
of which should go through the same walk-forward backtest before being
trusted.

Full per-game detail from this run is saved at
`data/processed/backtest_2023_2025.csv` so you can slice it any way you
like (pandas, Excel, whatever).

## Odds sources

Two are wired up:

1. **Free, no key (`HistoricalOddsSource`).** Used automatically when no
   API key is set. Comes from nflverse's public dataset, which includes
   the historical closing line for every game -- what powers the
   backtest above -- and this week's line, once the sportsbook market
   opens it. It's a single consensus-ish number, not a per-book
   breakdown, so `n_books` is fixed at 1 and the market-quality
   guardrails (min books / consensus gap / market width) are
   automatically skipped rather than failing every game on a
   technicality.

2. **Live multi-book (`LiveOddsAPI`).** Wraps
   [the-odds-api.com](https://the-odds-api.com/) (free tier: 500
   requests/month, plenty for one pull a week). Set the `ODDS_API_KEY`
   environment variable to enable it -- once set, `card` automatically
   uses it and you get real best-price/consensus/market-width numbers
   across the whitelisted books in `config.py`. **Note:** the client
   matches games by team name and the-odds-api returns full team names
   (e.g. "Kansas City Chiefs") while nflverse uses abbreviations (e.g.
   "KC") -- `odds.py` has a documented TODO where a name crosswalk needs
   to be added before this path is used for real; it's stubbed rather
   than silently wrong.

## Guardrails and Kelly sizing

Every constant from the original brief lives in `config.py`
(`GuardrailConfig`, `KellyConfig`) -- nothing is hardcoded elsewhere.
Change a number, rerun the backtest, compare. That's the loop this whole
project is built around: **collect -> measure -> test -> validate ->
adjust**, never "change something because of one good or bad week."

## Project layout

```
nfl_edge/
  config.py       guardrail/Kelly/training-window constants, all in one place
  data.py         loads historical games + embedded closing lines (free, no key)
  elo.py          walk-forward Elo rating engine (no lookahead by construction)
  qb.py           walk-forward QB rating (a proxy factor, honestly labeled as one)
  model.py        Elo -> calibration (Platt by default) -> p_home/p_home_cal/p_adj/p_final
  factors.py      multi-factor logistic model: Elo + QB + weather + home-field +
                   rest advantage + divisional game + offensive/defensive EPA +
                   sack-rate allowed/generated + travel distance/time-zone shift,
                   with data-fit base weights and slider-style multipliers on top
  spread.py       the same twelve factors via linear regression -> predicted margin ->
                   cover probability -> EV, for against-the-spread picks
  epa.py          walk-forward offensive/defensive EPA-per-play from real play-by-play
                   data (not opponent-adjusted -- see docs/DESIGN.md)
  pressure.py     walk-forward sack-rate allowed/generated from the same play-by-play
                   data -- a pressure proxy, honestly labeled as one (see docs/DESIGN.md)
  travel.py       travel distance + time-zone shift per game -- fixed geography, no
                   walk-forward state needed (see docs/DESIGN.md)
  stadiums.py     stadium_id -> (lat, lon) lookup, used for weather forecasts and travel.py
  weather_forecast.py  live Open-Meteo forecast for upcoming outdoor games (see below
                   for the network caveat that matters most)
  tracker.py      locks in each week's moneyline/spread pick (default weights, before
                   kickoff, never revised) and grades it once the game is final --
                   the page's actual track record, distinct from the backtest (see
                   docs/DESIGN.md)
  export_client_data.py  builds the JSON the interactive tuner page embeds
  odds.py         live multi-book client + free single-line fallback
  ev.py           fair-ML conversion, EV, guardrails, Kelly sizing
  card.py         builds the weekly "every game, both sides evaluated" card
  db.py           SQLite store: append-only card snapshots + grades
  grading.py      win%/ROI/units cuts, five-baseline comparison, live grading
  backtest.py     walk-forward backtest of the base (Elo-only) model
  cli.py          the weekly workflow as command-line subcommands, incl.
                   multifactor-backtest and spread-backtest
docs/
  DESIGN.md       the reasoning behind each module, and what to test next
data/
  raw/            cached downloads
  processed/      backtest exports + client_data.json (regenerated every run) +
                   tracked_predictions.json (NOT regenerated -- the locked, graded
                   track record; keep this file, don't delete/gitignore it)
  nfl_edge.sqlite  the results database
build/
  tuner_published.html  the last version published as the interactive tuner page
tuner_template.html  the tuner page template (has a __DATA_JSON__ placeholder --
                   see "The interactive tuner" below for how it gets built)
build_tuner.py    rebuilds build/tuner_published.html from the template + fresh JSON
weekly_refresh.sh  refresh-data + export_client_data + build_tuner.py in one command
dashboard.html    the static backtest-findings dashboard (published separately)
```

## The interactive tuner

Beyond the CLI, there's a shareable web page: **NFL Edge Tuner**,
published as a Claude Artifact -- a hosted page with its own URL that
lives on Anthropic's servers under this account, not on your PC, so it
opens the same way from any device without this project or conversation
needing to be open. It's private by default; use the page's own Share
menu to hand the link to someone if you want them to try it. Updating it
later (a new factor, a bug fix, fresh weekly data) republishes to the
*same* URL, so a link you've already shared just shows the new version
next time it's opened.

It adds eleven adjustable factors on top of Elo -- QB rating differential,
weather severity, extra home-field emphasis, rest advantage,
divisional-game closeness, offensive/defensive EPA-per-play
differentials (real play-by-play efficiency, not just win/loss record),
sack-rate allowed/generated differentials (protection and pass rush --
the closest free public data gets to pressure), and travel distance /
time-zone shift differentials (how much farther, and how many more zones,
the away team crossed to be there) -- each as a slider from 1
(ignore it) to 5 (lean on it twice as hard as the data suggested), with 3
anchored to whatever the data actually fit. Drag a slider and every pick for the
upcoming week, and the full 855-game backtest, recomputes instantly in
the browser -- no server, no reinstall, just a link you (or anyone) can
open and try. Every pick is tagged `HOME` or `AWAY` next to the team
name so you can see at a glance which side of the matchup the model
likes.

**Moneyline and spread, side by side.** A second toggle next to "This
week's picks / Full backtest" switches between the two: **Moneyline**
(who wins, the original view) and **Spread** (who covers, using a
margin-of-victory model against the real spread line and spread odds
for each game). The same eleven sliders drive both -- they're two separate
regressions under the hood (a win-probability logistic fit for
moneyline, a margin linear regression for spread), and the confidence
text in the left panel updates to describe whichever one is currently
selected, since the data says something different about each factor in
each model. One honest wrinkle: the "extra home-field emphasis" slider
has zero effect specifically on spread picks -- that's not a bug, it's
a real property of linear regression with a constant term (see
`docs/DESIGN.md`) -- and the page says so rather than pretending the
slider does something it can't.

**Track record.** A third tab, alongside "This week's picks" and "Full
backtest," shows how the page has actually done, not how a backtest
says it *would* have done. Every game's moneyline pick and spread pick
is locked in the first moment it appears as an upcoming game -- always
at default (1x) weights, regardless of wherever you've dragged the
sliders elsewhere on the page -- and is never revised afterward, even
once the model has seen more data. Once nflverse's data shows a final
score, the locked pick gets graded against it, and this tab shows a
running moneyline record and against-the-spread record (with pushes/ties
tracked separately, not counted as losses), plus a row of week pills you
can click through to see each week's picks, results, and whether each one
hit. It starts tracking from whenever this feature first shipped --
it does not backfill 2023-2025, since the "Full backtest" tab already
covers that history honestly under any slider setting. See
`docs/DESIGN.md` for the two fairness rules (lock-once, default-weights)
that make this a real track record instead of a hindsight report.

Each week's games also get a **confidence rank**, for real confidence
pools or a "pick 'em"/pick-5 style office game: moneyline and spread
picks are each ranked 1 (closest to a coin flip) through that week's
game count (most lopsided call), independently of each other, using
nothing but the model's own pick probability -- shown in small text
under each rank if you'd rather read the raw percentage. Same
lock-once rule applies: the ranking is set the moment that week's
games are locked and never reshuffled afterward, so it's still valid
the day you actually fill out a pool sheet with it. The table sorts
most-confident-first by default.

**Weather uses a live forecast now, not a placeholder** -- Open-Meteo,
free, no API key, for any upcoming outdoor game within its ~16-day
forecast window. A note right under the picks table says exactly how
many games got a real forecast on the current run vs. fell back to
neutral, so "weather did nothing this week" is never confused with
"the weather happened to be calm everywhere." The fallback is common in
one specific place worth knowing about: **this project's own cloud
scheduled refresh runs in an environment that blocks outbound requests
to general APIs like Open-Meteo**, so its weekly-automated runs will
show 0 live forecasts every time until that's run somewhere without
that restriction (your own machine, most CI runners, etc.) -- see
`docs/DESIGN.md` for the full explanation. The forecast fetch itself is
verified correct (tested offline against a mocked response, since even
this project's own dev environment can't reach the live API to test
against it directly); it's specifically *automatic weekly* forecasting
from the cloud schedule that's blocked, not the feature itself.

**A bug that shipped in earlier versions, now fixed:** a naming
mismatch between the exported data and the page's JavaScript meant the
weather and home-field sliders silently had *zero* effect on any number
on the page -- not "small," actually zero, on every pick and every
backtest row, in every version published before this fix. See
`docs/DESIGN.md` for the full story; both sliders now measurably move
the numbers, confirmed with a before/after check.

And yes: **the EV column is the confidence signal**, on both tabs. It's
the model's expected profit per dollar staked at the best available
price (moneyline) or spread price (spread), and a pick is flagged `YES`
to bet only once that edge clears the guardrail threshold for its
situation (bigger favorite, bigger underdog, and near-coinflip games
each require a bit more edge -- see `config.py`).

This is genuinely useful for exploring, and it's honest about its own
limits: the page's numbers use weights fit **once** on the whole window
(fast enough to feel instant), not the week-by-week walk-forward refit
the CLI's backtest uses elsewhere. If a slider setting looks good on the
page, confirm it for real with:

```bash
python -m nfl_edge.cli multifactor-backtest --qb 1.5 --weather 0.5 --home 1.0 --rest 1.0 --div 1.0 --epa-off 1.0 --epa-def 1.0 --sack-allowed 1.0 --sack-generated 1.0 --travel-distance 1.0 --travel-tz 1.0
python -m nfl_edge.cli spread-backtest --qb 1.5 --weather 0.5 --rest 1.0 --div 1.0 --epa-off 1.0 --epa-def 1.0 --sack-allowed 1.0 --sack-generated 1.0 --travel-distance 1.0 --travel-tz 1.0
```

which rerun that exact configuration the rigorous way (weights refit
weekly, using only games known before each prediction) and report real
win%/ROI/units. Regenerate the page's data after any model change or
once more of the season is played:

```bash
bash weekly_refresh.sh   # refresh-data + export_client_data + build_tuner.py, in order
```

(or run those three steps individually -- `weekly_refresh.sh` is just
the sequence spelled out; see it for the exact commands) then republish
`build/tuner_published.html`.

### What the sliders are honestly worth (2023-2025 fit)

Team strength (Elo), offensive EPA/play differential, sack-rate-allowed
differential (protection), travel-distance differential, and travel
time-zone-shift differential are the factors with a 90% bootstrap
confidence interval that stays clearly on one side of zero *somewhere* --
travel distance and sack-rate-allowed clear zero in the moneyline model,
travel time-zone-shift clears zero in the spread model instead, each
factor earning its keep in a different one of the two models. QB,
weather, extra home-field, rest advantage, divisional-game closeness,
defensive EPA/play, and sack-rate-generated (pass rush) all have intervals
that cross zero in both models -- meaning at 855 games, the data can't yet
confidently say these help *or* hurt (rest advantage comes closest,
leaning clearly positive, worth watching as more seasons accumulate).
That's not a reason the sliders shouldn't exist; it's exactly why they do.
See `docs/DESIGN.md` for the QB feature's construction, a mechanical bug
it avoided, the weather/home-field key-mismatch bug that was fixed, EPA's/
sack-rate's/travel's own construction and limitations, and what a
walk-forward sweep across several multiplier settings actually found
(short version: turning QB emphasis up modestly helped in that sweep,
though the whole strategy was still net-negative in every configuration
tested -- the underdog-overconfidence problem described above dominates
all of these secondary factors). Adding EPA moved both models in the
right direction without flipping the conclusion: moneyline all-picks ROI
improved from -8.3% to -6.9%, and the spread model's walk-forward backtest
improved from 47.4% ATS / -9.2% ROI to 48.9% ATS / -6.4% ROI at default
weights. Adding sack rate on top moved both further the same direction:
moneyline to -6.6% ROI (win rate flat at 40.2% -- a small, mixed effect,
since only the protection side of the two sack factors actually carries
signal), and spread to 49.2% ATS / -5.7% ROI. Adding travel distance/
time-zone shift on top of that was the single biggest jump of any factor
added yet: moneyline to -5.0% ROI (win rate 40.2% -> 42.0%), spread to
49.6% ATS / -5.1% ROI. It's also the most counter-intuitive result in the
project so far -- both travel coefficients came out *negative*, meaning
more away-team travel/time-zone burden is associated with home doing
*worse*, the opposite of the popular "travel fatigue helps the home team"
assumption. Reported as found, not corrected to match expectation -- see
`docs/DESIGN.md` for the honest reasoning about why, and a decomposition
showing travel distance alone actually walk-forwards even better on the
spread model (-4.0% ROI / 50.2% ATS) than combined with time-zone shift.
Still worse than the 52.4% breakeven point either way, and a further sign
that this model's edge, wherever it disagrees most with the market, tends
to be the model being wrong rather than a real opportunity.

## A word on discipline (this is in the brief for a reason)

The original brief's most important principle, and the one worth
repeating here: don't change the model because of one unusual week, and
don't add a feature because it seemed to matter once. Every change
should go through `backtest`, on the full training window, before you
trust it. The CLI and the walk-forward backtest exist specifically to
make that easy enough that you'll actually do it.
