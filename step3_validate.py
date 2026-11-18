import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = sys.argv[1]
rng = np.random.default_rng(7)

b = pd.read_pickle(f'{OUT}/balls.pkl')
bat = pd.read_pickle(f'{OUT}/bat.pkl')
bowl = pd.read_pickle(f'{OUT}/bowl.pkl')
RB = pd.read_pickle(f'{OUT}/rank_bat.pkl')
RW = pd.read_pickle(f'{OUT}/rank_bowl.pkl')
sb = pd.read_pickle(f'{OUT}/stint_bat.pkl')
sw = pd.read_pickle(f'{OUT}/stint_bowl.pkl')


L = b[b.legal].copy()
L['pr'] = np.where(L.pressure_hi, 'High pressure', 'Ordinary')


def summ(g):
    return pd.Series({
        'balls': len(g),
        'runs_per_over': g.team_runs.mean() * 6,
        'wkt_per_100': g.team_wkt.mean() * 100,
        'dot_%': (g.team_runs == 0).mean() * 100,
        'boundary_%': ((g.bat_runs >= 4)).mean() * 100,
        'abs_WP_swing_%': g.wpa.abs().mean() * 100,
        'balls_left': g.balls_left.mean(),
        'wkts_lost': g.wkts_before.mean()})


t1 = L.groupby(['innings', 'pr']).apply(summ).round(2)
print(t1)
mix = pd.crosstab([L.innings, L.pr], L.phase, normalize='index').round(3) * 100
print(mix)

ch = L[L.innings == 2].copy()
ch['rrr_b'] = pd.cut(ch.rrr_minus_crr, [-99, -2, 0, 2, 4, 99], labels=['<-2', '-2..0', '0..2', '2..4', '>4'])
t2 = ch.groupby('rrr_b', observed=True).agg(balls=('pressure_hi', 'size'), pct_high=('pressure_hi', 'mean'))
t2['pct_high'] *= 100
print(t2.round(1))


L['ko'] = L.knock_out == 1
print(L.groupby('ko').pressure_hi.mean().round(3))

t1.to_csv(f'{OUT}/val_hi_vs_lo.csv')
mix.to_csv(f'{OUT}/val_phase_mix.csv')
t2.to_csv(f'{OUT}/val_rrr.csv')

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for inn, c in [(1, '#1f6feb'), (2, '#d9480f')]:
    s = L[L.innings == inn].groupby('overid').pressure_hi.mean() * 100
    ax[0].plot(s.index, s.values, marker='o', color=c, label=f'Innings {inn}')
ax[0].set_xlabel('Over')
ax[0].set_ylabel('% balls high-pressure')
ax[0].legend()
ax[0].set_title('Where pressure occurs')
x = np.arange(2)
for i, inn in enumerate([1, 2]):
    ax[1].bar(x + i * 0.35, [t1.loc[(inn, p), 'abs_WP_swing_%'] for p in ['Ordinary', 'High pressure']], 0.35, label=f'Innings {inn}')
ax[1].set_xticks(x + 0.17)
ax[1].set_xticklabels(['Ordinary', 'High pressure'])
ax[1].set_title('Mean |win-prob swing| per ball (%)')
ax[1].legend()
ax[2].bar(t2.index.astype(str), t2.pct_high, color='#d9480f')
ax[2].set_xlabel('Required RR minus current RR')
ax[2].set_title('% high-pressure, chases')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_pressure.png', dpi=160)
plt.close()


def agg(s, w=None):

    s = s.copy()
    s['w'] = 1.0 if w is None else s.match_id.map(w).fillna(0)
    s['rw'] = s.res * s.w
    s['nw'] = s.n * s.w
    s['pw'] = (s.res > 0) * s.w
    s['sw'] = s.w
    hi = s[s.pr == 'HI'].groupby('pid')[['nw', 'rw', 'pw', 'sw']].sum()
    lo = s[s.pr == 'LO'].groupby('pid')[['nw', 'rw']].sum()
    return hi, lo


def rescore(s, elig, kA, kL, w, wts=None):
    hi, lo = agg(s, wts)
    hi = hi.reindex(elig)
    lo = lo.reindex(elig).fillna(0)
    A = (hi.rw / np.maximum(hi.nw, 1)) * hi.nw / (hi.nw + kA)
    Lv = (lo.rw / np.maximum(lo.nw, 1)) * lo.nw / (lo.nw + kL)
    cons = (hi.pw + 1) / (hi.sw + 2)
    z = lambda x: (x - x.mean()) / x.std()
    return w[0] * z(A) + w[1] * z(A - Lv) + w[2] * z(cons)


def kL(R, sig_lo):
    d = R
    v = max(np.var(d.L_raw, ddof=1) - np.mean(sig_lo / d.n_lo), 1e-6)
    return sig_lo / v


sg = json.load(open(f'{OUT}/sig.json'))
res = {}
for nm, R, s, sh, sl in [('bat', RB, sb, sg['sig_b_hi'], sg['sig_b_lo']), ('bowl', RW, sw, sg['sig_w_hi'], sg['sig_w_lo'])]:
    elig = R.index
    kA = R.kA.iloc[0]
    kl = kL(R, sl)
    W = (0.6, 0.15, 0.25)


    base = rescore(s, elig, kA, kl, W)
    assert np.allclose(base.sort_values(ascending=False).values, R.score.sort_values(ascending=False).values, atol=1e-6)


    mids = s.match_id.unique()
    ranks = []
    for _ in range(300):
        draw = pd.Series(rng.choice(mids, len(mids))).value_counts()
        sc = rescore(s, elig, kA, kl, W, draw)
        ranks.append(sc.rank(ascending=False))
    rk = pd.concat(ranks, axis=1)
    summ_ = pd.DataFrame({
        'rank_med': rk.median(axis=1),
        'rank_p5': rk.quantile(.05, axis=1),
        'rank_p95': rk.quantile(.95, axis=1),
        'p_top10': (rk <= 10).mean(axis=1)})


    grid = [(.6, .15, .25), (1, 0, 0), (.5, .25, .25), (.7, 0, .3), (.4, .2, .4), (.5, .5, 0), (.34, .33, .33), (.8, .1, .1)]
    top_base = set(R.index[:10])
    ov = {str(g): len(top_base & set(rescore(s, elig, kA, kl, g).sort_values(ascending=False).index[:10])) for g in grid}


    md = b.drop_duplicates('match_id').sort_values('match_date').match_id.tolist()
    half = {m: i % 2 for i, m in enumerate(md)}
    s['h'] = s.match_id.map(half)
    sc = []
    for h in (0, 1):
        sub = s[s.h == h]
        hi, _ = agg(sub)
        sc.append(hi.rw / hi.nw)
    sh_ = pd.concat(sc, axis=1, keys=['h0', 'h1']).dropna()
    sh_ = sh_.loc[sh_.index.intersection(elig)]

    nh = [agg(s[s.h == h])[0].nw for h in (0, 1)]
    ok = (nh[0].reindex(sh_.index) >= 40) & (nh[1].reindex(sh_.index) >= 40)
    r_sp = spearmanr(sh_[ok].h0, sh_[ok].h1)[0]
    r_p = np.corrcoef(sh_[ok].h0, sh_[ok].h1)[0, 1]


    wp = (R.w_hi / R.n_hi)
    cc = spearmanr(R.A_raw, wp)[0]
    cc2 = spearmanr(R.A, R.A_raw)[0]

    res[nm] = dict(summ=summ_, ov=ov, split_spearman=r_sp, split_pearson=r_p, split_n=int(ok.sum()), wpa_spearman=cc)

    out = R[['name', 'n_hi', 'matches', 'stints', 'A_raw', 'A', 'L', 'delta', 'cons', 'score']].join(summ_)
    out['rank'] = np.arange(1, len(out) + 1)
    out.to_csv(f'{OUT}/ranking_{nm}.csv')
    print(nm, 'sens overlap', ov, 'split', round(r_sp, 3), round(r_p, 3), ok.sum(), 'wpa corr', round(cc, 3))
    print(out.head(15)[['name', 'n_hi', 'A_raw', 'A', 'delta', 'cons', 'score', 'rank_p5', 'rank_med', 'rank_p95', 'p_top10']].round(3).to_string())

json.dump({k: {kk: (vv if not isinstance(vv, pd.DataFrame) else None) for kk, vv in v.items()} for k, v in res.items()},
          open(f'{OUT}/robust.json', 'w'), default=float)
