"""
dwc_utils.py - shared helper functions for the DataWorks Challenge 2026 notebooks.

Every notebook does `import dwc_utils as du` and calls the functions below.
Each function has a short explanation of WHAT it does and WHY.

Sections
  1. Paths and file finding
  2. Reading the Excel files (production, parameters, grid)
  3. Putting curves on a common quarterly time grid
  4. Proxy models (PCA + Gaussian Process, and two challengers)
  5. History matching (misfit)
  6. Forecasting beyond 2008 (liquid-rate decline x water-cut)
  7. Economics (NPV)
"""
import os
import re
import gc
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Canonical short names we use everywhere in the code
PARAMS = ["fault_trans", "poro_mult", "perm_mult", "aquifer_pv"]
PARAM_LABELS = {"fault_trans": "Fault Transmissibility", "poro_mult": "Porosity Multiplier",
                "perm_mult": "Permeability Multiplier", "aquifer_pv": "Aquifer Pore Volume"}
CUMS = ["oil_cum", "gas_cum", "water_cum"]
UNITS = {"oil_cum": "STB", "gas_cum": "MSCF", "water_cum": "STB",
         "oil_rate": "STB/d", "gas_rate": "MSCF/d", "water_rate": "STB/d"}
TEMPLATE_COLUMNS = {"gas_cum": "Gas production cumulative [MSCF]",
                    "oil_cum": "Oil production cumulative [STB]",
                    "water_cum": "Water production cumulative [STB]"}


# =====================================================================================
# 1. Paths and file finding
# =====================================================================================
def get_paths(base):
    """Return a dict of the project folders and create the output folders if missing."""
    p = {"BASE": base, "RAW": f"{base}/raw", "PROC": f"{base}/processed",
         "OUT": f"{base}/outputs", "FIG": f"{base}/outputs/figures", "PBI": f"{base}/powerbi"}
    for k in ["PROC", "OUT", "FIG", "PBI"]:
        os.makedirs(p[k], exist_ok=True)
    os.makedirs(f"{p['PROC']}/grid_features", exist_ok=True)
    os.makedirs(f"{p['PROC']}/grid_full", exist_ok=True)
    return p


def find_raw_files(raw_dir):
    """Find the 9 competition files by their number prefix (01 ... 09).
    Matching by number means extra spaces in a file name do not matter."""
    files = {}
    for f in sorted(os.listdir(raw_dir)):
        m = re.match(r"\s*(\d{2})", f)
        if m and f.lower().endswith((".xlsx", ".xlsm", ".xls")):
            files[m.group(1)] = os.path.join(raw_dir, f)
    missing = [f"{i:02d}" for i in range(1, 10) if f"{i:02d}" not in files]
    if missing:
        print("WARNING - these files were not found in raw/:", missing)
    return files


def require(path, made_by):
    """Stop with a clear message if a file from an earlier notebook is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{os.path.basename(path)}' not found. Run {made_by} first.")


# =====================================================================================
# 2. Reading the Excel files
# =====================================================================================
def to_dates(s):
    """Turn a column of dates into real dates.
    The competition files use US format MM/DD/YYYY (e.g. 11/21/1998), which a
    Malaysian computer could misread as day/month - so we give the format explicitly."""
    s = pd.Series(s)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    d = pd.to_datetime(s.astype(str).str.strip(), format="%m/%d/%Y", errors="coerce")
    other = pd.to_datetime(s, errors="coerce")          # real Excel date cells
    return d.fillna(other)


def classify_variable(text):
    """Map a column header like 'Oil production cumulative [STB]' to 'oil_cum'."""
    t = str(text).lower()
    fluid = "oil" if "oil" in t else "gas" if "gas" in t else "water" if "water" in t else None
    kind = "cum" if "cumul" in t else "rate" if "rate" in t else None
    return f"{fluid}_{kind}" if fluid and kind else None


def read_block_sheet(path, sheet):
    """Read one sheet laid out as: a 'Date' column + one column per series, with 1-3
    header rows above the data (e.g. 'X/Y1' row, 'Date/Case_1' row, variable-name row).
    The 'Date' heading may be in any column within the first 30 rows.
    Returns a long table: header, date, value  (blank cells removed), or None if the
    sheet is empty or has no 'Date' heading."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None, engine="calamine")
    if raw.empty or raw.shape[1] < 2:
        return None                                      # empty sheet (or chart-only sheet)
    top = raw.iloc[:30].astype(str).apply(lambda col: col.str.strip().str.lower())
    hits = np.argwhere(top.values == "date")             # (row, column) of every 'date' cell
    if len(hits) == 0:
        return None                                      # not a production sheet
    hdr, c0 = int(hits[0][0]), int(hits[0][1])           # first 'Date' cell: its row and column
    dates = to_dates(raw.iloc[:, c0]).reset_index(drop=True)
    data_rows = [i for i in range(hdr + 1, len(raw)) if pd.notna(dates.iloc[i])]
    if not data_rows:
        return None
    start = data_rows[0]
    # combine every header cell above the data into one label per column
    labels = []
    for c in range(c0 + 1, raw.shape[1]):
        parts = [str(raw.iat[r, c]).strip() for r in range(0, start)
                 if pd.notna(raw.iat[r, c]) and not re.fullmatch(r"Y\d+", str(raw.iat[r, c]).strip())]
        labels.append(" | ".join(parts) if parts else f"column_{c}")
    df = raw.iloc[start:, c0 + 1:].copy()
    df.columns = labels
    df.insert(0, "date", dates.iloc[start:].values)
    long = df.melt(id_vars="date", var_name="header", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["date", "value"])


def load_production(path02):
    """Read every sheet of file 02. Columns whose header contains 'Case_<n>' are
    simulated cases; all other columns are the observed (real field) history.
    Returns (sim, obs) as long tables."""
    xl = pd.ExcelFile(path02, engine="calamine")
    sims, obss = [], []
    for sh in xl.sheet_names:
        try:
            long = read_block_sheet(path02, sh)
        except Exception as e:                           # an unusual sheet must not stop everything
            print(f"  sheet '{sh}': skipped, could not be read ({type(e).__name__}: {e})")
            continue
        if long is None or long.empty:
            print(f"  sheet '{sh}': skipped (empty, or no Date column found)")
            continue
        long["var"] = long["header"].map(classify_variable)
        long["sheet"] = sh
        case = long["header"].str.extract(r"Case[_\s]*(\d+)", flags=re.I)[0]
        long["case_id"] = pd.to_numeric(case, errors="coerce")
        s, o = long[long.case_id.notna()], long[long.case_id.isna()]
        print(f"  sheet '{sh}': {s.case_id.nunique()} simulated cases, "
              f"{o.header.nunique()} observed columns, variables {sorted(long['var'].dropna().unique())}")
        sims.append(s); obss.append(o)
    sim = pd.concat(sims, ignore_index=True).dropna(subset=["var"])
    sim["case_id"] = sim["case_id"].astype(int)
    # several simulator report steps can share one date -> keep the last value of that day
    sim = (sim.sort_values(["case_id", "var", "date"])
              .groupby(["case_id", "var", "date"], as_index=False)["value"].last())
    obs = pd.concat(obss, ignore_index=True).dropna(subset=["var"]) if obss else pd.DataFrame()
    if len(obs):
        obs = obs.sort_values(["var", "date"]).groupby(["var", "date"], as_index=False)["value"].last()
    else:
        obs = pd.DataFrame({"var": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                            "value": pd.Series(dtype=float)})
        print("  WARNING: no real-field (observed) columns were found in file 02 - "
              "send a screenshot of the real-field sheet's top rows")
    return sim, obs


def load_parameters(path01):
    """Find the table that lists the 4 uncertain parameters per case in file 01."""
    xl = pd.ExcelFile(path01, engine="calamine")
    for sh in xl.sheet_names:
        raw = pd.read_excel(path01, sheet_name=sh, header=None, engine="calamine")
        mask = raw.apply(lambda col: col.astype(str).str.contains("Fault Transmissibility", case=False)).values
        if not mask.any():
            continue
        r, c0 = np.argwhere(mask)[0]
        header = raw.iloc[r].astype(str).str.strip()
        table = raw.iloc[r + 1:].copy(); table.columns = header
        item_col = header.iloc[0] if c0 > 0 else header.iloc[c0]
        rename = {}
        for col in header:
            low = col.lower()
            if "fault" in low: rename[col] = "fault_trans"
            elif "poro" in low: rename[col] = "poro_mult"
            elif "perm" in low: rename[col] = "perm_mult"
            elif "aquifer" in low: rename[col] = "aquifer_pv"
            elif low in ("item", "case", "case name", "name"): rename[col] = "item"
        table = table.rename(columns=rename)
        if "item" not in table.columns:
            table = table.rename(columns={item_col: "item"})
        table = table[table["item"].astype(str).str.contains(r"\d", na=False)]
        table["case_id"] = table["item"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        out = table[["case_id"] + PARAMS].apply(pd.to_numeric, errors="coerce").dropna()
        out["case_id"] = out["case_id"].astype(int)
        out["split"] = np.where(out.case_id <= 70, "train", np.where(out.case_id <= 85, "validation", "test"))
        print(f"  parameter table found in sheet '{sh}': {len(out)} cases")
        return out.sort_values("case_id").reset_index(drop=True)
    raise ValueError("Could not find a table with 'Fault Transmissibility' in file 01.")


# ---- grid files --------------------------------------------------------------------
NAME_FALLBACK = {"pore volume": "PORV", "porosity": "PORO", "permeability i": "PERMX", "permeability x": "PERMX",
                 "permeability j": "PERMY", "permeability y": "PERMY", "permeability k": "PERMZ",
                 "permeability z": "PERMZ", "net-to-gross": "NTG", "net to gross": "NTG",
                 "transmissibility i": "TRANX", "transmissibility x": "TRANX", "transmissibility j": "TRANY",
                 "transmissibility y": "TRANY", "transmissibility k": "TRANZ", "transmissibility z": "TRANZ",
                 "pressure": "PRESSURE", "water saturation": "SWAT", "oil saturation": "SOIL",
                 "gas saturation": "SGAS", "i index": "I", "j index": "J", "k index": "K",
                 "x coord": "X", "y coord": "Y", "z coord": "Z"}


def short_name(col):
    """'Porosity (PORO)' -> 'PORO'; 'Permeability K (PERMK)' -> 'PERMZ'; 'I Index' -> 'I'."""
    m = re.search(r"\(([A-Za-z0-9]+)\)", str(col))
    if m:
        code = m.group(1).upper()
        return {"PERMI": "PERMX", "PERMJ": "PERMY", "PERMK": "PERMZ"}.get(code, code)
    low = str(col).lower()
    for key, code in NAME_FALLBACK.items():
        if low.startswith(key):
            return code
    return str(col).strip().upper()


def grid_features_from_table(g, case_id):
    """Summarise one case's grid (thousands of cells) into a few numbers."""
    g = g[(g["PORV"] > 0)] if "PORV" in g else g
    f = {"case_id": case_id, "n_active_cells": int(len(g))}
    if "PORV" in g: f["pore_volume_total"] = float(g.PORV.sum())
    if {"PORV", "SOIL"} <= set(g.columns): f["oil_pore_volume"] = float((g.PORV * g.SOIL).sum())
    if {"PORV", "SGAS"} <= set(g.columns): f["gas_pore_volume"] = float((g.PORV * g.SGAS).sum())
    if {"PORV", "SWAT"} <= set(g.columns): f["water_pore_volume"] = float((g.PORV * g.SWAT).sum())
    if {"PORV", "PORO"} <= set(g.columns): f["porosity_mean"] = float(np.average(g.PORO, weights=g.PORV))
    if "PERMX" in g: f["permx_geomean"] = float(np.exp(np.log(g.PERMX.clip(lower=1e-6)).mean()))
    for c in ["PRESSURE", "SOIL", "SWAT"]:
        if c in g: f[f"{c.lower()}_mean"] = float(g[c].mean())
    return f


def convert_grid_file(path, proc_dir, keep_full=(1, 2)):
    """Read a big grid workbook one sheet (one case) at a time.
    - saves a small summary (features) for every case
    - saves the full cell table only for the cases in keep_full (for maps and checks)
    - skips cases already done, so you can simply re-run after a disconnect."""
    xl = pd.ExcelFile(path, engine="calamine")
    for sh in xl.sheet_names:
        nums = re.findall(r"\d+", sh)
        if not nums:
            print(f"  sheet '{sh}': no case number in name, skipped"); continue
        cid = int(nums[-1])
        fpath = f"{proc_dir}/grid_features/case_{cid:03d}.json"
        if os.path.exists(fpath):
            print(f"  case {cid:3d}: already done, skipped"); continue
        df = xl.parse(sh)
        df.columns = [short_name(c) for c in df.columns]
        df = df.apply(pd.to_numeric, errors="coerce").replace(-999.25, np.nan).astype("float32")
        feats = grid_features_from_table(df, cid)
        with open(fpath, "w") as fh:
            json.dump(feats, fh)
        if cid in keep_full:
            df.to_parquet(f"{proc_dir}/grid_full/case_{cid:03d}.parquet", index=False)
        print(f"  case {cid:3d}: {len(df):,} cells, columns: {', '.join(df.columns[:12])}{' ...' if df.shape[1] > 12 else ''}")
        del df; gc.collect()


def collect_grid_features(proc_dir):
    rows = []
    folder = f"{proc_dir}/grid_features"
    for f in sorted(os.listdir(folder)):
        with open(f"{folder}/{f}") as fh:
            rows.append(json.load(fh))
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True) if rows else pd.DataFrame()


# =====================================================================================
# 3. Common quarterly time grid
# =====================================================================================
def day_number(dates):
    """Dates -> days since 1970 (a plain number, easy to interpolate)."""
    return pd.to_datetime(pd.Series(dates)).values.astype("datetime64[D]").astype(float)


def quarterly_grid(end):
    return pd.date_range("1998-01-01", end, freq="QS")          # QS = first day of each quarter


def curves_on_grid(sim, case_ids, var, grid):
    """For each case: take its cumulative curve and read it off at every quarter date.
    Cumulative curves only go up, so straight-line interpolation is safe."""
    gx = day_number(grid)
    out = np.zeros((len(case_ids), len(grid)))
    sv = sim[sim["var"] == var]
    groups = {cid: g for cid, g in sv.groupby("case_id")}
    for i, cid in enumerate(case_ids):
        g = groups[cid].sort_values("date")
        out[i] = np.interp(gx, day_number(g["date"]), g["value"].values, left=0.0)
    return out


def obs_on_grid(obs, var, grid):
    """Observed cumulative read off at the quarter dates (NaN outside the observed period)."""
    o = obs[obs["var"] == var].sort_values("date")
    gx, ox = day_number(grid), day_number(o["date"])
    y = np.interp(gx, ox, o["value"].values, left=np.nan, right=np.nan)
    y[(gx < ox.min()) | (gx > ox.max())] = np.nan
    return y


def rates_from_cum(gx, cum):
    """Average daily rate in each quarter = change in cumulative / number of days."""
    return np.diff(cum, axis=-1) / np.diff(gx)


# =====================================================================================
# 4. Proxy models
# =====================================================================================
def scale_inputs(X, lo=None, hi=None):
    """Put the 4 parameters on a 0-1 scale. Permeability multiplier is log-transformed
    first because its effect is closer to 'per doubling' than 'per unit'."""
    X = np.asarray(X, float).copy()
    X[:, 2] = np.log10(X[:, 2])
    if lo is None:
        lo, hi = X.min(0), X.max(0)
    return (X - lo) / (hi - lo), lo, hi


def unscale_inputs(Xs, lo, hi):
    X = lo + np.asarray(Xs, float) * (hi - lo)
    X[..., 2] = 10 ** X[..., 2]
    return X


class PCAProxy:
    """Proxy = PCA (squeeze each curve into a few numbers) + a regressor for each number.
    method: 'gp' (Gaussian Process, our main model), 'poly' (polynomial ridge), 'forest' (random forest)."""

    def __init__(self, method="gp", var_keep=0.9999):
        self.method, self.var_keep = method, var_keep

    def _regressor(self, d):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import RidgeCV
        from sklearn.ensemble import RandomForestRegressor
        if self.method == "gp":
            k = (ConstantKernel(1.0) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
                 + WhiteKernel(1e-6, (1e-10, 1e-2)))
            return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=5, random_state=0)
        if self.method == "poly":
            return make_pipeline(PolynomialFeatures(3), RidgeCV(alphas=np.logspace(-6, 2, 30)))
        if self.method == "forest":
            return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=0)
        raise ValueError(self.method)

    def fit(self, Xs, Y):
        from sklearn.decomposition import PCA
        self.scale = float(np.nanmax(Y)) or 1.0
        self.pca = PCA(n_components=self.var_keep, svd_solver="full").fit(Y / self.scale)
        Z = self.pca.transform(Y / self.scale)
        self.models = [self._regressor(Xs.shape[1]).fit(Xs, Z[:, j]) for j in range(Z.shape[1])]
        return self

    def predict(self, Xs):
        Xs = np.atleast_2d(Xs)
        Z = np.column_stack([m.predict(Xs) for m in self.models])
        Y = self.pca.inverse_transform(Z) * self.scale
        return np.maximum.accumulate(np.clip(Y, 0, None), axis=1)   # cumulative never decreases


def score(y_true, y_pred):
    """RMSE, MAPE at the final date (%), and R2 over all dates."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    end = np.maximum(np.abs(y_true[:, -1]), 1e-9)
    mape = float(np.mean(np.abs(y_pred[:, -1] - y_true[:, -1]) / end) * 100)
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean(0)) ** 2))
    return rmse, mape, r2


# =====================================================================================
# 5. History matching
# =====================================================================================
class Misfit:
    """Misfit J = average of squared (predicted - observed) / sigma, summed over the chosen series.
    Series: the cumulative curves and the quarterly rates derived from them.
    sigma = assumed measurement error, as a fraction of each series' largest observed value."""

    def __init__(self, gx, observed, weights, sigma_frac=0.03):
        self.gx, self.w = gx, weights
        self.obs = {}
        for name, wgt in weights.items():
            if wgt <= 0:
                continue
            base, kind = name.rsplit("_", 1)
            y = observed[f"{base}_cum"]
            if kind == "rate":
                y = rates_from_cum(gx, y)
            self.obs[name] = y
        self.sig = {k: sigma_frac * np.nanmax(np.abs(v)) for k, v in self.obs.items()}

    def __call__(self, curves):
        """curves: dict like {'oil_cum': array (n_candidates, n_quarters), ...} -> misfit per candidate."""
        J = 0.0
        for name, y in self.obs.items():
            base, kind = name.rsplit("_", 1)
            c = curves[f"{base}_cum"]
            if kind == "rate":
                c = rates_from_cum(self.gx, c)
            ok = ~np.isnan(y)
            J = J + self.w[name] * np.mean(((c[:, ok] - y[ok]) / self.sig[name]) ** 2, axis=1)
        return J


# =====================================================================================
# 6. Forecasting beyond 2008
# =====================================================================================
def arps(t, qi, Di, b):
    """Arps hyperbolic decline: rate after t years of decline. b=0 ~ exponential, b=1 harmonic."""
    return qi / (1 + b * Di * t) ** (1 / b)


def logistic(Np, fmax, mid, width):
    """Water cut as an S-curve of cumulative oil Np: starts near 0, rises, levels off at fmax."""
    return fmax / (1 + np.exp(-(Np - mid) / width))


def decline_start(gx, oil, drop=0.97):
    """Day the plateau ends = first quarter whose oil rate is 3% below the plateau rate."""
    qo = rates_from_cum(gx, oil)
    peak = int(np.argmax(qo))
    after = np.where((qo < drop * qo[peak]) & (np.arange(len(qo)) > peak))[0]
    return gx[1:][after[0]] if len(after) else gx[-1]


def fit_forecaster(gx, oil, water, t_cut=None, b_max=1.0):
    """Fit (1) an Arps decline to the LIQUID rate (oil + water) after the plateau and
    (2) a logistic water-cut curve versus cumulative oil, using history up to t_cut.
    Returns forecast(days) -> (oil_cum, water_cum) and the fitted numbers."""
    from scipy.optimize import curve_fit
    if t_cut is not None:
        m = gx <= t_cut
        gx, oil, water = gx[m], oil[m], water[m]
    qo, qw = rates_from_cum(gx, oil), rates_from_cum(gx, water)
    qL = qo + qw
    tmid = (gx[:-1] + gx[1:]) / 2
    td = decline_start(gx, oil)
    d = tmid >= td
    if d.sum() < 6 or (qw > 1).sum() < 6:
        raise RuntimeError("fewer than 6 quarters of decline or water data")
    b_lo = min(0.01, b_max * 0.5)
    pL, _ = curve_fit(arps, (tmid[d] - td) / 365.25, qL[d], p0=[qL[d][0], 0.1, min(0.5, b_max * 0.9)],
                      bounds=([0, 1e-4, b_lo], [np.inf, 5, b_max]), maxfev=20000)
    fw = qw / np.maximum(qL, 1e-9)
    Npm = (oil[:-1] + oil[1:]) / 2
    w = qw > 1
    pW, _ = curve_fit(logistic, Npm[w], fw[w], p0=[min(0.95, fw[w].max() + 0.1), np.median(Npm[w]), Npm[w].std() + 1],
                      bounds=([fw[w].max(), 0, 1], [0.99, 10 * oil[-1], 10 * oil[-1]]), maxfev=20000)

    def forecast(days_out):
        days = np.arange(gx[-1], np.max(days_out) + 1)
        Np, Wp, o, wv = oil[-1], water[-1], np.empty(len(days)), np.empty(len(days))
        for k, dd in enumerate(days):                    # step forward one day at a time
            o[k], wv[k] = Np, Wp
            ql = arps((dd - td) / 365.25, *pL)
            f = logistic(Np, *pW)
            Np += ql * (1 - f)
            Wp += ql * f
        return np.interp(days_out, days, o), np.interp(days_out, days, wv)

    info = dict(decline_start=str(np.datetime64(int(td), "D")), qi=pL[0], Di=pL[1], b=pL[2],
                wc_max=pW[0], wc_mid=pW[1], wc_width=pW[2])
    return forecast, info


def fit_oil_only_arps(gx, oil, t_cut=None):
    """Simpler reference method: Arps decline fitted directly to the OIL rate."""
    from scipy.optimize import curve_fit
    if t_cut is not None:
        m = gx <= t_cut
        gx, oil = gx[m], oil[m]
    qo = rates_from_cum(gx, oil)
    tmid = (gx[:-1] + gx[1:]) / 2
    td = decline_start(gx, oil)
    d = tmid >= td
    if d.sum() < 6:
        raise RuntimeError("fewer than 6 quarters of decline")
    p, _ = curve_fit(arps, (tmid[d] - td) / 365.25, qo[d], p0=[qo[d][0], 0.2, 0.5],
                     bounds=([0, 1e-4, 0.01], [np.inf, 5, 1.0]), maxfev=20000)

    def forecast(days_out):
        days = np.arange(gx[-1], np.max(days_out) + 1)
        c = oil[-1] + np.concatenate([[0], np.cumsum(arps((days[:-1] - td) / 365.25, *p))])
        return np.interp(days_out, days, c)
    return forecast


# =====================================================================================
# 7. Economics
# =====================================================================================
def yearly_cashflow(dates, oil_cum, gas_cum, water_cum, oil_price, gas_price, water_cost, opex_per_year,
                    discount_rate, start_year=2008):
    """Turn cumulative forecasts into a yearly cash-flow table and discount it (NPV)."""
    df = pd.DataFrame({"date": pd.to_datetime(dates), "oil": oil_cum, "gas": gas_cum, "water": water_cum})
    df["year"] = df.date.dt.year
    year_end = df.groupby("year").last()[["oil", "gas", "water"]]
    prev = year_end.shift(1)
    prev.iloc[0] = df[["oil", "gas", "water"]].iloc[0].values      # first row = start of forecast
    vol = (year_end - prev).clip(lower=0)
    cf = pd.DataFrame({"oil_stb": vol.oil, "gas_mscf": vol.gas, "water_stb": vol.water})
    cf["revenue_usd"] = cf.oil_stb * oil_price + cf.gas_mscf * gas_price
    cf["cost_usd"] = cf.water_stb * water_cost + opex_per_year
    cf["net_cash_usd"] = cf.revenue_usd - cf.cost_usd
    cf["year_index"] = cf.index - start_year
    cf["discounted_usd"] = cf.net_cash_usd / (1 + discount_rate) ** cf.year_index
    return cf.reset_index()
