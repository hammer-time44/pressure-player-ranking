import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

D, OUT = sys.argv[1], sys.argv[2]

b = pd.read_csv(f'{D}/ball_by_ball.csv', keep_default_na=False, na_values=[''])
m = pd.read_csv(f'{D}/matches.csv')
c = pd.read_csv(f'{D}/competitions.csv')
m = m.merge(c[['comp_id', 'comp_name', 'match_format']], on='comp_id')

FULL = {'India', 'Pakistan', 'West Indies', 'New Zealand', 'Bangladesh', 'England',
        'Sri Lanka', 'Australia', 'South Africa', 'Zimbabwe', 'Afghanistan', 'Ireland'}


keep = m.comp_name.str.startswith('IPL') | ((m.match_format == 'T20I') & m.home_team.isin(FULL) & m.away_team.isin(FULL))

keep &= m.revised_target.isna() & m.first_inn_revised_overs.isna() & m.second_inn_revised_overs.isna()
m = m[keep].copy()
m['comp'] = np.where(m.comp_name.str.startswith('IPL'), 'IPL', 'T20I')

b = b[b.match_id.isin(m.match_id) & b.innings.isin([1, 2])].copy()
b = b.merge(m[['match_id', 'comp', 'win_team_id', 'knock_out', 'stage', 'match_date']], on='match_id')
b = b.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)


ex = b.extras_type.fillna('None')
b['legal'] = ~ex.isin(['wide', 'Overstep', 'other', 'Penalty'])
b['bat_runs'] = b.runs
b['team_runs'] = b.runs + b.extras1_runs + b.extras_2_runs
b['bowl_runs'] = b.runs + np.where(ex.isin(['wide', 'Overstep', 'other']), b.extras1_runs + b.extras_2_runs, 0)
b['faced'] = ~ex.isin(['wide', 'Penalty'])

dt = b.dismissal_type.fillna('')
b['team_wkt'] = b.is_wicket.astype(int)
b['bowler_wkt'] = (b.is_wicket & dt.isin(['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket'])).astype(int)
b['bat_out'] = (b.is_wicket & (b.dismissed_player_id == b.striker_id) & (dt != 'Retire Out')).astype(int)


g = b.groupby(['match_id', 'innings'])
b['score_before'] = g.team_runs.cumsum() - b.team_runs
b['wkts_before'] = g.team_wkt.cumsum() - b.team_wkt
b['balls_before'] = g.legal.cumsum() - b.legal.astype(int)
b['balls_left'] = 120 - b.balls_before

tot = g.team_runs.sum().rename('inn_total').reset_index()
b = b.merge(tot, on=['match_id', 'innings'])
t1 = tot[tot.innings == 1].set_index('match_id').inn_total
b['target'] = np.where(b.innings == 2, b.match_id.map(t1) + 1, np.nan)
b = b[(b.balls_left > 0) & (b.wkts_before < 10)].copy()


b['bat_win'] = (b.batting_team_id == b.win_team_id).astype(int)
b = b[b.win_team_id.notna()].copy()

b['phase'] = pd.cut(b.overid, [0, 6, 15, 20], labels=['PP', 'MID', 'DEATH']).astype(str)
b['comp_t20i'] = (b.comp == 'T20I').astype(int)
b['rem_runs'] = b.inn_total - b.score_before


ch = b.innings == 2
b['need'] = np.where(ch, b.target - b.score_before, np.nan)
b['rrr'] = b.need / (b.balls_left / 6)
b['crr'] = b.score_before / np.maximum(b.balls_before, 1) * 6
F1 = ['balls_left', 'wkts_before', 'score_before', 'comp_t20i']
F2 = ['balls_left', 'wkts_before', 'need', 'rrr', 'comp_t20i']
b['rrr_minus_crr'] = np.where(ch, b.rrr - b.crr, np.nan)


Rf = ['balls_left', 'wkts_before', 'innings', 'comp_t20i']
Rm = HistGradientBoostingRegressor(max_iter=200, learning_rate=.08, max_depth=4, random_state=0).fit(b[Rf], b.rem_runs)
b['R_cur'] = Rm.predict(b[Rf])


def R_at(balls_left, wk):
    X = pd.DataFrame({'balls_left': balls_left, 'wkts_before': np.minimum(wk, 10),
                      'innings': b.innings.values, 'comp_t20i': b.comp_t20i.values})
    return np.where(balls_left <= 0, 0, Rm.predict(X)) * (wk < 10)


bl_next = b.balls_left - b.legal.astype(int)
b['R_next_nowk'] = R_at(bl_next.values, b.wkts_before.values)
b['R_next_bat'] = R_at(bl_next.values, (b.wkts_before + b.bat_out).values)
b['R_next_bowl'] = R_at(bl_next.values, (b.wkts_before + b.bowler_wkt).values)


b['wp'] = np.nan


def fitwp(df, feat):
    p = np.zeros(len(df))
    gkf = GroupKFold(5)
    for tr, te in gkf.split(df, groups=df.match_id):
        mdl = HistGradientBoostingClassifier(max_iter=150, learning_rate=.06, max_depth=4, min_samples_leaf=200, random_state=0)
        mdl.fit(df.iloc[tr][feat], df.iloc[tr].bat_win)
        p[te] = mdl.predict_proba(df.iloc[te][feat])[:, 1]
    full = HistGradientBoostingClassifier(max_iter=150, learning_rate=.06, max_depth=4, min_samples_leaf=200, random_state=0).fit(df[feat], df.bat_win)
    return p, full


i1 = b.index[b.innings == 1]
i2 = b.index[b.innings == 2]
b.loc[i1, 'wp'], m1 = fitwp(b.loc[i1], F1)
b.loc[i2, 'wp'], m2 = fitwp(b.loc[i2], F2)


outs = [('0', 0, 0), ('1', 1, 0), ('2', 2, 0), ('3', 3, 0), ('4', 4, 0), ('6', 6, 0), ('W', 0, 1)]
lg = b[b.legal]
mix = {}
for ph, gp in lg.groupby('phase'):
    p = np.array([
        (gp.team_runs == 0) & (gp.team_wkt == 0),
        (gp.team_runs == 1) & (gp.team_wkt == 0),
        (gp.team_runs == 2) & (gp.team_wkt == 0),
        (gp.team_runs == 3) & (gp.team_wkt == 0),
        (gp.team_runs == 4) & (gp.team_wkt == 0),
        (gp.team_runs >= 5) & (gp.team_wkt == 0),
        (gp.team_wkt == 1),
    ]).mean(axis=1)
    mix[ph] = p / p.sum()


def wp_after(df, model, inn, r, w):
    d = pd.DataFrame(index=df.index)
    d['balls_left'] = df.balls_left - 1
    d['wkts_before'] = df.wkts_before + w
    d['comp_t20i'] = df.comp_t20i
    if inn == 1:
        d['score_before'] = df.score_before + r
        X = d[F1]
    else:
        d['need'] = df.need - r
        d['rrr'] = np.where(d.balls_left > 0, d.need / np.maximum(d.balls_left / 6, 1 / 6), 99)
        X = d[F2]
    p = model.predict_proba(X)[:, 1]

    if inn == 2:
        p = np.where(d.need <= 0, 1.0, p)
        p = np.where((d.balls_left <= 0) & (d.need > 0), 0.0, p)
        p = np.where(d.wkts_before >= 10, np.where(d.need > 0, 0.0, 1.0), p)
    return p


b['lev'] = np.nan
for inn, idx, model in [(1, i1, m1), (2, i2, m2)]:
    sub = b.loc[idx]
    acc = np.zeros(len(sub))
    P = np.vstack([sub.phase.map(lambda x: mix[x][k]).values for k in range(len(outs))])
    for k, (_, r, w) in enumerate(outs):
        a = wp_after(sub, model, inn, r, w)
        acc += P[k] * np.abs(a - sub.wp.values)
    b.loc[idx, 'lev'] = acc


b['lev_pct'] = b.groupby('innings').lev.rank(pct=True)
b['pressure_hi'] = (b.lev_pct >= 0.75)


b['wp_next'] = b.groupby(['match_id', 'innings']).wp.shift(-1)
b['wpa'] = np.where(b.wp_next.notna(), b.wp_next - b.wp, np.nan)

b.to_pickle(f'{OUT}/balls.pkl')
print(b.shape, b.pressure_hi.mean())


for inn in (1, 2):
    s = b[b.innings == inn]
    s = s.assign(bin=pd.cut(s.wp, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]))
    print(inn, s.groupby('bin', observed=True).agg(pred=('wp', 'mean'), act=('bat_win', 'mean'), n=('wp', 'size')).round(3).to_dict('list'))
for inn in (1, 2):
    s = b[b.innings == inn]
    print('AUC', inn, roc_auc_score(s.bat_win, s.wp))
