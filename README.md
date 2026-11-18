# pressure-player-ranking

T20 pressure reliability score. Ranks batters and bowlers by how well they perform in high-pressure situations, adjusted for context and corrected for small samples.

## Setup

```
pip install numpy pandas scikit-learn scipy matplotlib
```

Data dir must contain `ball_by_ball.csv`, `matches.csv`, `competitions.csv`.

## Run

```
python step1_state.py <data_dir> <out_dir>
python step2_score.py <out_dir>
python step3_validate.py <out_dir>
python step4_report_data.py <out_dir>
```

Run in order; each step reads the previous step's pickles from `<out_dir>`.

## Pipeline

1. **State and pressure** (`step1_state.py`)
   - Filters to IPL and full-member T20Is, drops rain-affected and no-result matches.
   - Builds per-ball state: score, wickets, balls left, target, required rate.
   - Fits a run-expectancy model and a separate win-probability model per innings (out-of-fold by match).
   - Pressure (leverage) = expected |win-prob change| on the next ball. The top quartile within each innings is flagged high-pressure.

2. **Scoring** (`step2_score.py`)
   - Ball value = runs plus change in expected remaining runs (wickets cost the batting side).
   - Each ball is compared to the average for its context (innings, phase, pressure, competition, pace/spin).
   - Per-player pressure rating `A`, ordinary rating `L`, `delta = A - L`, and consistency `cons` (share of pressure stints above peers).
   - Empirical-Bayes shrinkage on `A` and `L`.
   - Final score = 0.6·z(A) + 0.15·z(delta) + 0.25·z(cons), with minimum-ball and minimum-stint cutoffs.

3. **Validation** (`step3_validate.py`)
   - Checks high-pressure balls differ from ordinary ones (wickets, dots, win-prob swing, phase mix, chase rate gap, knockouts).
   - Bootstrap over matches gives rank intervals and `p_top10`.
   - Weight-sensitivity grid on top-10 overlap, split-half reliability, and a win-prob-added cross-check.
   - Writes final `out/ranking_bat.csv` and `out/ranking_bowl.csv`.

4. **Report data** (`step4_report_data.py`)
   - Top-10 tables, league baselines and charts.

## Outputs

| File | Content |
|---|---|
| `ranking_bat.csv`, `ranking_bowl.csv` | Full rankings with score, components, rank intervals, `p_top10` |
| `top10_*.csv`, `league.json` | Top-10 stats vs league baseline |
| `fig_*.png` | Pressure validation, top-10 bars, pressure vs ordinary scatter |
