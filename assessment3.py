import os
import math
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

#   Paths 
DATA1 = "dataset1.csv"
DATA2 = "dataset2.csv"
OUT_DIR = "Results"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

#   Utilities  
def parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def winsorize_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):
        return series
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k*iqr, q3 + k*iqr
    return series.clip(lower, upper)

def adjusted_r2(r2: float, n: int, p: int) -> float:
    return float("nan") if (n - p - 1) == 0 else 1 - (1 - r2) * (n - 1) / (n - p - 1)

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    X_num = X.select_dtypes(include=[np.number]).copy()
    X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(X_num.median(numeric_only=True))
    Xc = sm.add_constant(X_num, has_constant="add")
    rows = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        rows.append({"Variable": col, "VIF": variance_inflation_factor(Xc.values, i)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)

def residual_plots(y_true: pd.Series, y_pred: np.ndarray, prefix: str):
    resid = y_true - y_pred

    # Residuals vs Fitted
    plt.figure()
    plt.scatter(y_pred, resid, alpha=0.6)
    plt.axhline(0.0)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Fitted — {prefix}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{prefix}_residuals_vs_fitted.png"))
    plt.close()

    # Histogram
    plt.figure()
    plt.hist(resid.dropna(), bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(f"Residuals Histogram — {prefix}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{prefix}_residuals_hist.png"))
    plt.close()

    # Q-Q
    plt.figure()
    stats.probplot(resid.dropna(), dist="norm", plot=plt)
    plt.title(f"Q-Q Plot — {prefix}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{prefix}_qq.png"))
    plt.close()

def cv_rmse(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Tuple[float, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses = []
    for tr, te in kf.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        mse = mean_squared_error(y.iloc[te], pred)  # ← no 'squared' kw
        rmse = float(np.sqrt(mse))
        rmses.append(rmse)
    return float(np.mean(rmses)), float(np.std(rmses))

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    return df

def save_table(df: pd.DataFrame, name: str):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

#   Load & Merge  
d1 = pd.read_csv(DATA1)
d2 = pd.read_csv(DATA2)

# Parse times
d1["start_time"] = parse_dt(d1["start_time"])
d2["time"] = parse_dt(d2["time"])

# Drop and sort by timestamp
d1 = d1.dropna(subset=["start_time"]).sort_values("start_time").reset_index(drop=True)
d2 = d2.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

# Rolling windows on d2 (assuming ~30-min interval: windows 2=1h, 4=2h)
d2 = d2.set_index("time").sort_index()
for w in [2, 4]:
    d2[f"rat_minutes_roll_mean_{w}"] = d2["rat_minutes"].rolling(window=w, min_periods=1).mean()
    d2[f"rat_arrivals_roll_sum_{w}"] = d2["rat_arrival_number"].rolling(window=w, min_periods=1).sum()
    d2[f"bat_landing_roll_mean_{w}"] = d2["bat_landing_number"].rolling(window=w, min_periods=1).mean()
    d2[f"food_availability_roll_mean_{w}"] = d2["food_availability"].rolling(window=w, min_periods=1).mean()
d2 = d2.reset_index()

# Nearest merge within 30 minutes
merged = pd.merge_asof(
    d1, d2,
    left_on="start_time", right_on="time",
    direction="nearest", tolerance=pd.Timedelta("30min")
)

# Target & features
target = "bat_landing_to_food"
features = [
    "seconds_after_rat_arrival","risk","reward",
    "hours_after_sunset_x" if "hours_after_sunset_x" in merged.columns else "hours_after_sunset",
    "season",
    "rat_minutes","rat_arrival_number","bat_landing_number","food_availability",
    "rat_minutes_roll_mean_2","rat_arrivals_roll_sum_2",
    "bat_landing_roll_mean_2","food_availability_roll_mean_2",
    "rat_minutes_roll_mean_4","rat_arrivals_roll_sum_4"
]

# Harmonise column names
rename = {}
if "hours_after_sunset_x" in merged.columns:
    rename["hours_after_sunset_x"] = "hours_after_sunset"
if "month_x" in merged.columns:
    rename["month_x"] = "month"
merged = merged.rename(columns=rename)

# Keep available
features = [c for c in features if c in merged.columns]

# Drop rows without target
df = merged.dropna(subset=[target]).copy()

# Winsorize numeric columns
for col in features + [target]:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        df[col] = winsorize_iqr(df[col])

# Impute missing
for col in features:
    if col in df.columns and df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

# One-hot season
if "season" in df.columns and not pd.api.types.is_numeric_dtype(df["season"]):
    df = pd.get_dummies(df, columns=["season"], drop_first=True)

# Build X, y
X_cols = [c for c in df.columns if c in features or c.startswith("season_")]
X = ensure_numeric(df[X_cols].copy())
y = df[target].astype(float)

#   OLS (statsmodels): Hypothesis Tests  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ols = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
print(ols.summary())

# Save coefficient table with t, p
coef_table = ols.summary2().tables[1].reset_index().rename(columns={"index":"Variable"})
save_table(coef_table, "ols_coefficients_train.csv")

# Residual diagnostics (train and test)
y_train_pred = ols.predict(sm.add_constant(X_train, has_constant="add"))
residual_plots(y_train, y_train_pred, "ols_train")

# Test predictions/metrics
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)
adj = adjusted_r2(r2, len(y_test), X_test.shape[1])
metrics_overall = pd.DataFrame([{"MAE":mae,"MSE":mse,"RMSE":rmse,"R2":r2,"AdjR2":adj}])
save_table(metrics_overall, "metrics_overall.csv")
residual_plots(y_test, y_pred, "ols_test")

# Multicollinearity
vif_df = compute_vif(X_train)
save_table(vif_df, "vif.csv")

#   Cross-Validation (5-fold RMSE)  
cv_model = LinearRegression()
cv_mean, cv_std = cv_rmse(cv_model, X, y, n_splits=5)
cv_df = pd.DataFrame([{"CV_kfold":5,"RMSE_mean":cv_mean,"RMSE_std":cv_std}])
save_table(cv_df, "cv_rmse.csv")

#   Regularisation: Ridge & Lasso  
alphas = [0.01, 0.1, 1.0, 10.0]
rows = []

for name, Est in [("Ridge", Ridge), ("Lasso", Lasso)]:
    for a in alphas:
        pipe = Pipeline([
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("reg", Est(alpha=a, max_iter=10000))
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        _rmse = float(np.sqrt(mse))
        _r2 = r2_score(y_test, pred)
        _adj = adjusted_r2(_r2, len(y_test), X_test.shape[1])

        rows.append({
            "Model": name, "alpha": a,
            "RMSE": _rmse, "R2": _r2, "AdjR2": _adj
        })
    

reg_df = pd.DataFrame(rows).sort_values(["Model","RMSE"])
save_table(reg_df, "regularisation_results.csv")

#   Standardised Coefficients (importance)  
Xtr_num = X_train.select_dtypes(include=[np.number]).copy()
for c in Xtr_num.columns:
    if Xtr_num[c].isna().any():
        Xtr_num[c] = Xtr_num[c].fillna(Xtr_num[c].median())
means = Xtr_num.mean()
stds = Xtr_num.std(ddof=0).replace(0, 1.0)
Xtr_std = (Xtr_num - means) / stds
lr_std = LinearRegression().fit(Xtr_std, y_train)
coef_std = pd.DataFrame({"Variable": Xtr_std.columns, "Standardized_Coefficient": lr_std.coef_}).sort_values(
    "Standardized_Coefficient", key=lambda s: s.abs(), ascending=False
)
save_table(coef_std, "standardized_coefficients.csv")

#   Seasonal Models (winter vs spring)  
def seasonal_metrics(df_in: pd.DataFrame, label: str) -> Dict[str, float]:
    if len(df_in) < 30:
        return {"Subset": label, "Note": "Too few rows"}
    Xs = ensure_numeric(df_in[X_cols].copy())
    ys = df_in[target].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    yp = m.predict(Xte)
    _mae = mean_absolute_error(yte, yp)
    _mse = mean_squared_error(yte, yp)
    _rmse = math.sqrt(_mse)
    _r2 = r2_score(yte, yp)
    _adj = adjusted_r2(_r2, len(yte), Xte.shape[1])
    return {"Subset": label, "MAE":_mae,"MSE":_mse,"RMSE":_rmse,"R2":_r2,"AdjR2":_adj}

season_rows = []
coef_rows = []

# If season is original categorical
if "season" in df.columns and df["season"].dtype == object:
    seasons = sorted(df["season"].dropna().unique().tolist())
    for s in seasons:
        sub = df[df["season"] == s].copy()
        season_rows.append(seasonal_metrics(sub, s))
        # per-season OLS coefficients (if enough rows)
        if len(sub) > len(X_cols) + 5:
            m = sm.OLS(sub[target].astype(float), sm.add_constant(ensure_numeric(sub[X_cols]), has_constant="add")).fit()
            for k, v in m.params.items():
                if k != "const":
                    coef_rows.append({"Variable": k, "Season": s, "Coefficient": v})
# If one-hot columns exist
else:
    for lab in ["winter","spring"]:
        col = f"season_{lab}"
        if col in df.columns:
            sub = df[df[col] == 1].copy()
            if len(sub) > 0:
                season_rows.append(seasonal_metrics(sub, lab))
                if len(sub) > len(X_cols) + 5:
                    m = sm.OLS(sub[target].astype(float), sm.add_constant(ensure_numeric(sub[X_cols]), has_constant="add")).fit()
                    for k, v in m.params.items():
                        if k != "const":
                            coef_rows.append({"Variable": k, "Season": lab, "Coefficient": v})

season_df = pd.DataFrame(season_rows)
save_table(season_df, "metrics_by_season.csv")

coef_compare_df = pd.DataFrame(coef_rows)
if not coef_compare_df.empty:
    save_table(coef_compare_df, "seasonal_coefficients_ols.csv")
    # Plot variables with largest coefficient variability
    top_vars = (
        coef_compare_df.groupby("Variable")["Coefficient"].std().sort_values(ascending=False).head(10).index.tolist()
    )
    for var in top_vars:
        sub = coef_compare_df[coef_compare_df["Variable"] == var]
        plt.figure()
        plt.bar(sub["Season"], sub["Coefficient"])
        plt.title(f"Seasonal Coefficients — {var}")
        plt.xlabel("Season")
        plt.ylabel("OLS Coefficient")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"seasonal_coef_{var}.png"))
        plt.close()

print("\\nAll outputs saved in:", OUT_DIR)
print("Figures in:", FIG_DIR)
