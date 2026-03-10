"""
FIFA Player Stats - ML Models for predictions.
Trains: Value predictor, Potential predictor, Similar players (KNN).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

FEATURE_COLS = ["Age", "Overall", "Potential_overall", "Wage", "Total_stats"]
KNN_FEATURE_COLS = ["Age", "Overall", "Total_stats", "Speed", "Power", "Passing", "Defense", "Shooting"]


def prepare_features(df: pd.DataFrame, stats_fn) -> tuple:
    """Prepare feature matrix and targets. Returns (X, y_value, y_potential, X_knn, player_stats, df)."""
    df = df.copy()
    # Ensure numeric columns
    for col in FEATURE_COLS + ["Value", "Wage"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + ["Value", "Potential_overall"])
    df = df[(df["Value"] > 0) & (df["Wage"] >= 0)]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)

    # Build derived stats for each player
    player_stats = {}
    stats_rows = []
    for _, row in df.iterrows():
        s = stats_fn(row)
        player_stats[row["Player_name"]] = s
        stats_rows.append(list(s.values()))

    stat_cols = list(player_stats[next(iter(player_stats))].keys())
    stats_df = pd.DataFrame(stats_rows, columns=stat_cols, index=df.index)
    df = pd.concat([df.reset_index(drop=True), stats_df], axis=1)

    X = df[FEATURE_COLS].astype(float)
    y_value = np.log1p(df["Value"].astype(float))
    y_potential = df["Potential_overall"].astype(float)
    # Drop any rows with NaN in targets
    valid = ~(y_value.isna() | y_potential.isna())
    X, y_value, y_potential = X[valid], y_value[valid], y_potential[valid]
    df = df[valid].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y_value = y_value.reset_index(drop=True)
    y_potential = y_potential.reset_index(drop=True)

    X_knn = df[KNN_FEATURE_COLS].astype(float).replace([np.inf, -np.inf], np.nan)
    # Ensure no NaN for KNN
    knn_valid = X_knn.notna().all(axis=1)
    X_knn = X_knn[knn_valid].fillna(0)
    df = df[knn_valid].reset_index(drop=True)
    X = X[knn_valid].reset_index(drop=True)
    y_value = y_value[knn_valid].reset_index(drop=True)
    y_potential = y_potential[knn_valid].reset_index(drop=True)
    player_stats = {k: v for k, v in player_stats.items() if k in df["Player_name"].values}
    return X, y_value, y_potential, X_knn, player_stats, df


def train_models(df: pd.DataFrame, stats_fn) -> dict:
    """Train Value, Potential regressors and KNN for similar players. Returns model dict."""
    X, y_value, y_potential, X_knn, player_stats, df_clean = prepare_features(df, stats_fn)

    n = len(X)
    if n < 10:
        raise ValueError(
            f"Not enough samples to train models (got {n}, need at least 10). "
            "Ensure the dataset has valid Value, Wage, and other required columns."
        )

    scaler_value = StandardScaler()
    scaler_potential = StandardScaler()
    scaler_knn = StandardScaler()

    X_scaled = scaler_value.fit_transform(X)
    X_pot_scaled = scaler_potential.fit_transform(X[["Age", "Overall", "Total_stats"]])
    X_knn_scaled = scaler_knn.fit_transform(X_knn)

    X_tr, X_te, yv_tr, yv_te = train_test_split(X_scaled, y_value, test_size=0.2, random_state=42)
    Xptr, Xpte, yp_tr, yp_te = train_test_split(X_pot_scaled, y_potential, test_size=0.2, random_state=42)

    rf_value = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    rf_value.fit(X_tr, yv_tr)
    pred_v = rf_value.predict(X_te)
    mae_value = mean_absolute_error(yv_te, pred_v)
    r2_value = r2_score(yv_te, pred_v)

    rf_potential = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_potential.fit(Xptr, yp_tr)
    pred_p = rf_potential.predict(Xpte)
    mae_pot = mean_absolute_error(yp_te, pred_p)
    r2_pot = r2_score(yp_te, pred_p)

    knn = NearestNeighbors(n_neighbors=11, metric="euclidean")
    knn.fit(X_knn_scaled)

    return {
        "value_model": rf_value,
        "potential_model": rf_potential,
        "knn": knn,
        "scaler_value": scaler_value,
        "scaler_potential": scaler_potential,
        "scaler_knn": scaler_knn,
        "df_clean": df_clean,
        "player_stats": player_stats,
        "X_knn": X_knn,
        "X_knn_scaled": X_knn_scaled,
        "metrics": {
            "value_mae": mae_value,
            "value_r2": r2_value,
            "potential_mae": mae_pot,
            "potential_r2": r2_pot,
        },
    }


def predict_value(model_dict: dict, row: pd.Series) -> float:
    """Predict market value (in original scale)."""
    X = pd.DataFrame([{c: row[c] for c in FEATURE_COLS}], columns=FEATURE_COLS).astype(float)
    X_scaled = model_dict["scaler_value"].transform(X)
    log_val = model_dict["value_model"].predict(X_scaled)[0]
    return float(np.expm1(log_val))


def predict_potential(model_dict: dict, row: pd.Series) -> float:
    """Predict potential overall rating."""
    pot_cols = ["Age", "Overall", "Total_stats"]
    X = pd.DataFrame([{c: row[c] for c in pot_cols}], columns=pot_cols).astype(float)
    X_scaled = model_dict["scaler_potential"].transform(X)
    return float(model_dict["potential_model"].predict(X_scaled)[0])


def get_similar_players(model_dict: dict, player_name: str, exclude_self: bool = True, k: int = 6) -> list:
    """Return list of (player_name, distance) for k nearest players."""
    df = model_dict["df_clean"]
    stats = model_dict["player_stats"]
    knn = model_dict["knn"]
    X_scaled = model_dict["X_knn_scaled"]

    if player_name not in stats:
        return []
    row = df[df["Player_name"] == player_name].iloc[0]
    player_full = pd.DataFrame([{
        "Age": row["Age"], "Overall": row["Overall"], "Total_stats": row["Total_stats"],
        "Speed": stats[player_name]["Speed"], "Power": stats[player_name]["Power"],
        "Passing": stats[player_name]["Passing"], "Defense": stats[player_name]["Defense"],
        "Shooting": stats[player_name]["Shooting"]
    }], columns=KNN_FEATURE_COLS).astype(float)
    player_scaled = model_dict["scaler_knn"].transform(player_full)
    distances, indices = knn.kneighbors(player_scaled, n_neighbors=k + (1 if exclude_self else 0))
    out = []
    for i, idx in enumerate(indices[0]):
        name = df.iloc[idx]["Player_name"]
        if exclude_self and name == player_name:
            continue
        out.append((name, float(distances[0][i])))
    return out[:k]
