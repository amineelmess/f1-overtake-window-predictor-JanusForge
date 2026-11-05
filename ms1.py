"""Team JanusForge — <Owen, Joshua, Amine, Idris>
Milestone 1 — F1 overtake-window predictor: baseline vs. simple PyTorch model on the same held-out race.

Pipeline: (1) remove cars in the pits -> (2) keep close on-track pairs -> (3) predict a pass within H laps -> (4) check rain effect.
Run:  pip install -r requirements.txt && python milestone1.py     (downloads the data release into ./data on first run)
"""
import copy, io, urllib.request, zipfile
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import torch, torch.nn as nn
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, precision_score, recall_score)

URL = ("https://github.com/amineelmess/f1-overtake-window-predictor-JanusForge/"
       "releases/download/milestone1-data-v1/milestone1_data.zip")
DATA = Path("data")
SEED, H, GAP_MAX, DRS_GAP = 0, 3, 3.0, 1.0   # H: horizon (laps); GAP_MAX: "close together" (s); DRS_GAP: baseline rule (s)
K = ["session_key", "driver_number"]
torch.manual_seed(SEED); np.random.seed(SEED); Path("figures").mkdir(exist_ok=True)

if not list(DATA.rglob("laps.csv")):                      # download + unzip the dataset once
    try:
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read())).extractall(DATA)
    except Exception as e:
        raise SystemExit(f"Download failed ({e}). Download milestone1_data.zip from the GitHub release and unzip it into ./data/")

def rd(name):
    """read <name>.csv; if the release has one folder per race, concatenate them"""
    fs = [f for f in DATA.rglob(f"{name}.csv") if "__MACOSX" not in str(f)]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates()
T = lambda s: pd.to_datetime(s, utc=True, format="ISO8601").astype("datetime64[ns, UTC]")
def asof(left, right, cols, by=K, lk="t1"):
    """attach the latest `right` record (by `date`) at or before time `lk`, per `by` group"""
    r = right.rename(columns={"date": "_t"}).sort_values("_t")
    return pd.merge_asof(left.sort_values(lk), r[["_t", *by, *cols]], left_on=lk, right_on="_t", by=by).drop(columns="_t")

laps = rd("laps"); laps["t0"] = T(laps.date_start)
laps = laps.dropna(subset=["t0", "lap_duration"]).copy()
laps["t1"] = laps.t0 + pd.to_timedelta(laps.lap_duration, unit="s")            # lap end time
iv = rd("intervals"); iv["date"] = T(iv.date); iv["gap_to_leader"] = pd.to_numeric(iv.gap_to_leader, errors="coerce")
pos = rd("position"); pos["date"] = T(pos.date)
wx = rd("weather"); wx["date"] = T(wx.date)

df = asof(laps, iv, ["gap_to_leader"])                                         # NaN for lapped cars -> dropped below
df = asof(df, pos, ["position"])
df = asof(df, wx, ["rainfall", "track_temperature", "humidity"], by=["session_key"])
df = df.sort_values(K + ["lap_number"]).reset_index(drop=True)

# tyre state
st = rd("stints").sort_values("lap_start")
df = pd.merge_asof(df.sort_values("lap_number"), st[["lap_start", "compound", "tyre_age_at_start", *K]],
                   left_on="lap_number", right_on="lap_start", by=K)
df["tyre_age"] = df.tyre_age_at_start + df.lap_number - df.lap_start
df["compound"] = df.compound.map({"SOFT": 0, "MEDIUM": 1, "HARD": 2}).fillna(1)

# yellow flag / safety car on this lap
rc = rd("race_control")
cau = (rc[rc.flag.isin(["YELLOW", "DOUBLE YELLOW"]) | (rc.category == "SafetyCar")][["session_key", "lap_number"]]
       .drop_duplicates().assign(caution=1))
df = df.merge(cau, on=["session_key", "lap_number"], how="left").fillna({"caution": 0})

# STEP 1: take every car in the pits out (in-lap = pit entry lap, out-lap = lap after it)
pit = rd("pit"); pit_set = set(zip(pit.session_key, pit.driver_number, pit.lap_number))
df["in_pit"] = [(s, d, l) in pit_set or (s, d, l - 1) in pit_set or o
                for s, d, l, o in zip(df.session_key, df.driver_number, df.lap_number, df.is_pit_out_lap.astype(bool))]
n0 = len(df); df = df[~df.in_pit & df.position.notna() & df.gap_to_leader.notna()].copy()
print(f"driver-laps: {n0} -> {len(df)} after removing pit cars / lapped cars")

# STEP 2: nearest ON-TRACK car ahead (pit cars are gone, so they can't be the 'car ahead')
df = df.sort_values(["session_key", "lap_number", "position"]); g = df.groupby(["session_key", "lap_number"])
for c, n in [("driver_number", "def_drv"), ("gap_to_leader", "_gl"), ("lap_duration", "_dl"), ("tyre_age", "_dt"), ("compound", "def_compound")]:
    df[n] = g[c].shift()
df["gap_ahead"] = df.gap_to_leader - df._gl
df["pace_delta"] = (df._dl - df.lap_duration).clip(-5, 5)                       # >0: chaser was faster this lap
df["tyre_delta"] = df.tyre_age - df._dt                                         # >0: chaser's tyres are older
df = df.sort_values(K + ["lap_number"])
same = df.def_drv == df.groupby(K).def_drv.shift()                              # same car ahead as last lap?
df["gap_trend"] = np.where(same, df.groupby(K).gap_ahead.diff(), 0)             # <0: closing in
df["rain"] = (df.rainfall.fillna(0) > 0).astype(float)
df["lap_frac"] = df.lap_number / df.groupby("session_key").lap_number.transform("max")
ov = rd("overtakes"); ov["t"] = T(ov.date)
ov = pd.merge_asof(ov.sort_values("t"), laps[["t0", "lap_number", "session_key", "driver_number"]]
                   .rename(columns={"driver_number": "overtaking_driver_number"}).sort_values("t0"),
                   left_on="t", right_on="t0", by=["session_key", "overtaking_driver_number"]).dropna(subset=["lap_number"])
near_pit = lambda c: np.array([(s, d, l) in pit_set or (s, d, l - 1) in pit_set for s, d, l in zip(ov.session_key, ov[c], ov.lap_number)])
n_raw = len(ov)
ov = ov[(ov.lap_number > 1) & ~near_pit("overtaking_driver_number") & ~near_pit("overtaken_driver_number")]
print(f"overtake events: {n_raw} raw -> {len(ov)} on-track passes kept")

ev = (pd.concat([ov.assign(lap_number=ov.lap_number - h) for h in range(1, H + 1)])
      .groupby(["session_key", "overtaking_driver_number", "overtaken_driver_number", "lap_number"]).size().rename("n").reset_index()
      .rename(columns={"overtaking_driver_number": "driver_number", "overtaken_driver_number": "def_drv"}))
df = df.merge(ev, on=K + ["def_drv", "lap_number"], how="left"); df["y"] = (df.n.fillna(0) > 0).astype(int)

FEATS = ["gap_ahead", "gap_trend", "pace_delta", "tyre_age", "tyre_delta", "compound", "def_compound",
         "position", "caution", "track_temperature", "humidity", "rain", "lap_frac"]
df = df[(df.lap_number >= 2) & (df.lap_number < df.groupby(K).lap_number.transform("max"))
        & (df.position > 1) & (df.gap_ahead > 0) & (df.gap_ahead <= GAP_MAX)].copy()      # STEP 2: close pairs only
df[FEATS] = df[FEATS].astype(float).fillna(0)

# Split by race, chronological: last race = test, the one before = validation, the rest = train (5 races -> 3/1/1)
start = laps.groupby("session_key").t0.min(); races = sorted(df.session_key.unique(), key=lambda k: start[k])
if len(races) >= 3:
    df["split"] = df.session_key.map(lambda k: "test" if k == races[-1] else "val" if k == races[-2] else "train"); how = "by race"
else:                                                                                       # fallback if <3 races are present
    df["split"] = np.select([df.lap_frac <= .60, df.lap_frac <= .75], ["train", "val"], "test"); how = "by lap"
tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])
print(f"{len(races)} race(s), {df.driver_number.nunique()} drivers, {len(df)} close-pair rows, split {how}; test race(s): "
      f"{sorted(te.session_key.unique())}")
print("input files (rows):", {f: len(rd(f)) for f in ["laps", "intervals", "position", "overtakes", "pit", "stints", "weather", "race_control"]})
print(pd.DataFrame({"rows": df.groupby("split").size(), "positives": df.groupby("split").y.sum(),
                    "pos_rate": df.groupby("split").y.mean().round(3)}).loc[["train", "val", "test"]])
# EDA: overtake probability vs gap, and the rain question (does rain suppress overtakes?)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
b = pd.cut(df.gap_ahead, [0, .5, 1, 1.5, 2, 3]); r = df.groupby(b, observed=True).y.mean()
ax[0].bar(r.index.astype(str), r.values); ax[0].set(xlabel="gap to car ahead (s)", ylabel=f"P(pass within {H} laps)", title="Close pairs")
rain_tbl = df.groupby("rain").y.agg(pairs="size", passes="sum", rate="mean").round(3)
if df.rain.nunique() > 1:
    ax[1].bar(["dry", "rain"], rain_tbl.rate.values); ax[1].set(ylabel=f"P(pass within {H} laps)", title="Rain vs dry (close pairs)")
else:
    ax[1].text(.5, .5, "no rain laps in these races", ha="center"); ax[1].axis("off")
plt.tight_layout(); plt.savefig("figures/eda.png", dpi=150); plt.close()
print(rain_tbl if df.rain.nunique() > 1 else "No rain in the loaded races: the rain effect cannot be estimated yet (feature is constant).")
print(df.groupby("session_key").agg(rain_share=("rain", "mean"), pairs=("y", "size"), pass_rate=("y", "mean")).round(3))
mu, sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)                    # standardise with train statistics only
tt = lambda d: torch.tensor(((d[FEATS] - mu) / sd).values, dtype=torch.float32)
Xtr, Xva, Xte = tt(tr), tt(va), tt(te)
ytr, yva = (torch.tensor(d.y.values, dtype=torch.float32) for d in (tr, va)); yte = te.y.values
F, EPOCHS, BS, LR = len(FEATS), 100, 128, 1e-2
net = lambda h: nn.Linear(F, 1) if h == 0 else nn.Sequential(nn.Linear(F, h), nn.ReLU(), nn.Linear(h, 1))
bce = nn.BCEWithLogitsLoss()

def fit(h, wd):
    torch.manual_seed(SEED); m = net(h); opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=wd)
    curve, best = [], (np.inf, None, 0)
    for e in range(EPOCHS):
        m.train()
        for idx in torch.randperm(len(Xtr)).split(BS):
            opt.zero_grad(); bce(m(Xtr[idx]).squeeze(1), ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): tl, vl = bce(m(Xtr).squeeze(1), ytr).item(), bce(m(Xva).squeeze(1), yva).item()
        curve.append((tl, vl))
        if vl < best[0]: best = (vl, copy.deepcopy(m.state_dict()), e + 1)
    m.load_state_dict(best[1]); return m, np.array(curve), best

runs = {(h, wd): fit(h, wd) for h in [0, 16] for wd in [0, 1e-3, 1e-2]}
(h_best, wd_best), (model, curve, best) = min(runs.items(), key=lambda kv: kv[1][2][0])
print(pd.DataFrame([(h, wd, r[2][2], round(r[2][0], 4)) for (h, wd), r in runs.items()],
                   columns=["hidden", "weight_decay", "best_epoch", "val_loss"]).to_string(index=False))
print(f"selected hidden={h_best}, weight_decay={wd_best}, epoch={best[2]}, parameters={sum(p.numel() for p in model.parameters())}")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for (h, wd), r in runs.items(): ax[0].plot(r[1][:, 1], label=f"h={h}, wd={wd:g}")
ax[0].set(xlabel="epoch", ylabel="validation loss", title="Hyper-parameter search"); ax[0].legend(fontsize=6)
ax[1].plot(curve[:, 0], label="train"); ax[1].plot(curve[:, 1], label="validation"); ax[1].axvline(best[2] - 1, ls="--", c="gray")
ax[1].set(xlabel="epoch", ylabel="BCE loss", title=f"Training curve (h={h_best}, wd={wd_best:g})"); ax[1].legend()
plt.tight_layout(); plt.savefig("figures/training_curve.png", dpi=150); plt.close()
if h_best == 0:
    print(pd.Series(model.weight.detach().numpy()[0], FEATS).round(3).sort_values())     # linear weights (standardised features)
with torch.no_grad(): pv, pt = (torch.sigmoid(model(X)).squeeze(1).numpy() for X in (Xva, Xte))
ts = np.unique(np.quantile(pv, np.linspace(.5, .99, 50))); yv = va.y.values
thr = ts[np.argmax([f1_score(yv, pv >= t) for t in ts])]
g, prior = te.gap_ahead.values, tr.y.mean()

def metrics(y, s, pred):
    return dict(PR_AUC=average_precision_score(y, s), ROC_AUC=roc_auc_score(y, s), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), F1=f1_score(y, pred))
res = pd.DataFrame({"prior (no skill)": metrics(yte, np.full(len(yte), prior), np.zeros(len(yte), bool)),
                    f"baseline: gap<{DRS_GAP}s": metrics(yte, -g, g < DRS_GAP),
                    "PyTorch model": metrics(yte, pt, pt >= thr)}).T.round(3)
res["params"] = [1, 1, sum(p.numel() for p in model.parameters())]

rng, ci = np.random.default_rng(SEED), {"baseline": [], "model": [], "model - baseline": []}
for _ in range(500):
    i = rng.integers(0, len(yte), len(yte))
    if yte[i].sum() == 0: continue
    a, b = average_precision_score(yte[i], -g[i]), average_precision_score(yte[i], pt[i])
    ci["baseline"].append(a); ci["model"].append(b); ci["model - baseline"].append(b - a)
print(f"test rows={len(yte)}, positives={yte.sum()}, threshold={thr:.3f}")
for k, v in ci.items(): print(f"PR-AUC {k:17s} 95% CI [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]")
print(res.to_string())
res.to_csv("figures/results.csv")