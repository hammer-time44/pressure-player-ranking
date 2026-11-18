import sys
import json
import numpy as np
import pandas as pd

OUT = sys.argv[1]
b = pd.read_pickle(f'{OUT}/balls.pkl')
b['pr'] = np.where(b.pressure_hi, 'HI', 'LO')


b['bat_val'] = b.bat_runs + b.R_next_bat - b.R_cur

b['bowl_val'] = -b.bowl_runs + (b.R_cur - b.R_next_bowl)
b['pace'] = b.pace_or_spin.fillna('Pace')

bat = b[b.faced].copy()
bowl = b.copy()


ctx_bat = ['innings', 'phase', 'pr', 'comp', 'pace']
ctx_bowl = ['innings', 'phase', 'pr', 'comp', 'pace']
bat['exp'] = bat.groupby(ctx_bat).bat_val.transform('mean')
bat['res'] = bat.bat_val - bat.exp
bowl['exp'] = bowl.groupby(ctx_bowl).bowl_val.transform('mean')
bowl['res'] = bowl.bowl_val - bowl.exp


bat['wres'] = bat.wpa
bowl['wres'] = -bowl.wpa


def stint(df, pid, pname):
    s = df.groupby([pid, 'match_id', 'innings', 'pr']).agg(
        n=('res', 'size'), res=('res', 'sum'), wres=('wres', 'sum'), name=(pname, 'first')).reset_index()
    return s.rename(columns={pid: 'pid'})


sb = stint(bat, 'striker_id', 'striker_name')
sw = stint(bowl, 'bowler_id', 'bowler_name')


def build(s, min_hi, min_lo, min_stints):
    hi = s[s.pr == 'HI'].groupby('pid').agg(
        n_hi=('n', 'sum'), res_hi=('res', 'sum'), w_hi=('wres', 'sum'), stints=('n', 'size'),
        name=('name', 'first'), pos=('res', lambda x: (x > 0).sum()), matches=('match_id', 'nunique'))
    lo = s[s.pr == 'LO'].groupby('pid').agg(n_lo=('n', 'sum'), res_lo=('res', 'sum'))
    d = hi.join(lo, how='left').fillna({'n_lo': 0, 'res_lo': 0})
    d = d[(d.n_hi >= min_hi) & (d.n_lo >= min_lo) & (d.stints >= min_stints)].copy()
    return d


def eb(mean, n, sig2_ball):

    vb = max(np.var(mean, ddof=1) - np.mean(sig2_ball / n), 1e-6)
    k = sig2_ball / vb
    return mean * n / (n + k), k


def score(df, sig_hi=None, sig_lo=None, w=(0.6, 0.15, 0.25)):
    d = df.copy()
    d['A_raw'] = d.res_hi / d.n_hi
    d['L_raw'] = d.res_lo / d.n_lo
    d['A'], d['kA'] = eb(d.A_raw.values, d.n_hi.values, sig_hi)
    d['L'], _ = eb(d.L_raw.values, d.n_lo.values, sig_lo)
    d['delta'] = d.A - d.L
    d['cons'] = (d.pos + 1) / (d.stints + 2)
    z = lambda x: (x - x.mean()) / x.std()
    d['score'] = w[0] * z(d.A) + w[1] * z(d.delta) + w[2] * z(d.cons)
    return d.sort_values('score', ascending=False)


sig_b_hi = bat[bat.pr == 'HI'].res.var()
sig_b_lo = bat[bat.pr == 'LO'].res.var()
sig_w_hi = bowl[bowl.pr == 'HI'].res.var()
sig_w_lo = bowl[bowl.pr == 'LO'].res.var()

BAT = build(sb, 150, 100, 20)
BOWL = build(sw, 120, 90, 20)
RB = score(BAT, sig_hi=sig_b_hi, sig_lo=sig_b_lo)
RW = score(BOWL, sig_hi=sig_w_hi, sig_lo=sig_w_lo)

print('eligible', len(BAT), len(BOWL))
pd.set_option('display.width', 250)
cols = ['name', 'n_hi', 'stints', 'A_raw', 'A', 'L', 'delta', 'cons', 'score']
print(RB[cols].head(15).round(3))
print(RW[cols].head(15).round(3))

RB.to_pickle(f'{OUT}/rank_bat.pkl')
RW.to_pickle(f'{OUT}/rank_bowl.pkl')
sb.to_pickle(f'{OUT}/stint_bat.pkl')
sw.to_pickle(f'{OUT}/stint_bowl.pkl')
bat.to_pickle(f'{OUT}/bat.pkl')
bowl.to_pickle(f'{OUT}/bowl.pkl')
json.dump(dict(sig_b_hi=sig_b_hi, sig_b_lo=sig_b_lo, sig_w_hi=sig_w_hi, sig_w_lo=sig_w_lo), open(f'{OUT}/sig.json', 'w'))
