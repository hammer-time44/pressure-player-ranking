import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1]
bat = pd.read_pickle(f'{OUT}/bat.pkl')
bowl = pd.read_pickle(f'{OUT}/bowl.pkl')
rb = pd.read_csv(f'{OUT}/ranking_bat.csv', index_col=0)
rw = pd.read_csv(f'{OUT}/ranking_bowl.csv', index_col=0)
tb = rb.head(10).copy()
tw = rw.head(10).copy()


def bstats(pid):
    d = bat[bat.striker_id == pid]
    r = {}
    for p in ('HI', 'LO'):
        x = d[d.pr == p]
        r[f'sr_{p}'] = x.bat_runs.sum() / len(x) * 100
        r[f'out_{p}'] = x.bat_out.sum() / max(len(x), 1) * 100
        r[f'bpo_{p}'] = len(x) / max(x.bat_out.sum(), 1)
    return r


def wstats(pid):
    d = bowl[bowl.bowler_id == pid]
    r = {}
    for p in ('HI', 'LO'):
        x = d[d.pr == p]
        lg = x.legal.sum()
        r[f'eco_{p}'] = x.bowl_runs.sum() / lg * 6
        r[f'wk_{p}'] = x.bowler_wkt.sum() / lg * 100
        r[f'dot_{p}'] = (x.bowl_runs == 0).mean() * 100
    return r


tb = tb.join(pd.DataFrame({i: bstats(i) for i in tb.index}).T)
tw = tw.join(pd.DataFrame({i: wstats(i) for i in tw.index}).T)


lb = {p: (bat[bat.pr == p].bat_runs.sum() / len(bat[bat.pr == p]) * 100, bat[bat.pr == p].bat_out.mean() * 100) for p in ('HI', 'LO')}
lw = {p: (bowl[bowl.pr == p].bowl_runs.sum() / bowl[bowl.pr == p].legal.sum() * 6,
          bowl[bowl.pr == p].bowler_wkt.sum() / bowl[bowl.pr == p].legal.sum() * 100) for p in ('HI', 'LO')}
print(lb, lw)

tb.to_csv(f'{OUT}/top10_bat.csv')
tw.to_csv(f'{OUT}/top10_bowl.csv')
json.dump(dict(lb=lb, lw=lw), open(f'{OUT}/league.json', 'w'))
print(tb[['name', 'sr_HI', 'out_HI', 'sr_LO', 'out_LO']].round(1))
print(tw[['name', 'eco_HI', 'wk_HI', 'eco_LO', 'wk_LO']].round(2))


for nm, t, col in [('bat', tb, '#1f6feb'), ('bowl', tw, '#d9480f')]:
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.barh(t.name, t.score, color=col, alpha=.85)
    ax.set_xlabel('Pressure Reliability Score')
    ax.set_title(f'Top 10 {"batters" if nm == "bat" else "bowlers"}')
    for i, (s, lo, hi_) in enumerate(zip(t.score, t.rank_p5, t.rank_p95)):
        ax.text(s + .03, i, f'rank 90% range {int(lo)}-{int(hi_)}', va='center', fontsize=7)
    ax.set_xlim(0, t.score.max() * 1.6)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_top_{nm}.png', dpi=170)
    plt.close()


for nm, R in [('bat', rb), ('bowl', rw)]:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(R.A, R.L, s=R.n_hi / 6, alpha=.45, color='#6b7280')
    top = R.head(10)
    ax.scatter(top.A, top.L, s=top.n_hi / 6, color='#d9480f', alpha=.85)
    for _, r in top.iterrows():
        ax.annotate(r['name'].split()[-1], (r.A, r.L), fontsize=7)
    ax.axhline(0, lw=.5, c='k')
    ax.axvline(0, lw=.5, c='k')
    ax.set_xlabel('Shrunk pressure rating (value per ball vs peers)')
    ax.set_ylabel('Shrunk ordinary-situation rating')
    ax.set_title(f'{"Batters" if nm == "bat" else "Bowlers"}: pressure vs usual (bubble = pressure balls)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_scatter_{nm}.png', dpi=170)
    plt.close()


b = pd.read_pickle(f'{OUT}/balls.pkl')
print(b.match_id.nunique(), len(b), b.comp.value_counts().to_dict(), b.groupby('comp').match_id.nunique().to_dict(), b.match_date.min(), b.match_date.max())
print(len(rb), len(rw))
print(rb.matches.describe()[['min', '50%']].to_dict(), tb[['n_hi', 'stints']].to_dict('list'), tw[['n_hi', 'stints']].to_dict('list'))
