import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json
import os
import warnings
from collections import defaultdict, deque

warnings.filterwarnings('ignore')

DATA_PATH   = "data/results.csv"
MODEL_DIR   = "model"
FORM_WINDOW = 10          # last N matches for form features
ELO_START   = 1500        # starting ELO for every team
ELO_D       = 400         # ELO scaling factor (standard)
EVAL_FROM   = "2018-01-01"  # use post-2018 as test set

# Tournament importance → ELO K-factor
K_MAP = {
    "FIFA World Cup":             60,
    "UEFA Euro":                  50,
    "Copa América":               50,
    "Africa Cup of Nations":      45,
    "AFC Asian Cup":              45,
    "CONCACAF Gold Cup":          40,
    "OFC Nations Cup":            40,
    "UEFA Nations League":        35,
    "Confederations Cup":         45,
    "Friendly":                   10,
}
K_DEFAULT = 30


def k_factor(tournament: str) -> int:
    for name, k in K_MAP.items():
        if name.lower() in tournament.lower():
            return k
    if "friendly" in tournament.lower():
        return 10
    return K_DEFAULT


def tournament_weight(tournament: str) -> float:
    """0–1 importance score used as a feature."""
    k = k_factor(tournament)
    return round(k / 60, 4)


# LOAD & CLEAN DATA

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_team","away_team","home_score","away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    # Outcome from home team perspective
    def outcome(row):
        if row["home_score"] > row["away_score"]: return "Win"
        if row["home_score"] < row["away_score"]: return "Loss"
        return "Draw"

    df["outcome"] = df.apply(outcome, axis=1)

    # Neutral ground flag
    if "neutral" not in df.columns:
        df["neutral"] = False
    df["is_neutral"] = df["neutral"].astype(int)

    # World Cup flag
    df["is_wc"] = df["tournament"].str.contains("World Cup", case=False, na=False).astype(int)

    # Tournament weight feature
    df["t_weight"] = df["tournament"].apply(tournament_weight)

    print(f"✅ Loaded {len(df):,} matches  |  "
          f"{df['date'].min().year}–{df['date'].max().year}  |  "
          f"{df['outcome'].value_counts().to_dict()}")
    return df


# ELO RATINGS

def expected_score(r_home, r_away):
    return 1.0 / (1.0 + 10 ** ((r_away - r_home) / ELO_D))


def goal_diff_multiplier(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    return 1.75 + (gd - 3) * 0.1 


def compute_elo(df: pd.DataFrame):
    elo = defaultdict(lambda: ELO_START)
    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        eh, ea = elo[h], elo[a]

        home_elos.append(eh)
        away_elos.append(ea)

        # Actual scores
        if row["outcome"] == "Win":
            s_h, s_a = 1.0, 0.0
        elif row["outcome"] == "Loss":
            s_h, s_a = 0.0, 1.0
        else:
            s_h = s_a = 0.5

        exp_h = expected_score(eh, ea)
        exp_a = 1.0 - exp_h
        K     = k_factor(row["tournament"])
        gdm   = goal_diff_multiplier(row["home_score"] - row["away_score"])

        elo[h] = eh + K * gdm * (s_h - exp_h)
        elo[a] = ea + K * gdm * (s_a - exp_a)

    df = df.copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    # Save final ELO snapshot for inference
    final_elo = dict(elo)
    print(f" ELO computed for {len(final_elo):,} teams")
    print(f"   Top 5: { sorted(final_elo.items(), key=lambda x:-x[1])[:5] }")
    return df, final_elo


# RECENT FORM FEATURES

def compute_form(df: pd.DataFrame):
    # team → deque of recent match dicts
    history = defaultdict(lambda: deque(maxlen=FORM_WINDOW))

    cols = {
        "home_form_win_rate":   [],
        "home_form_goal_avg":   [],
        "home_form_concede_avg":[],
        "home_form_pts5":       [],  # points from last 5 (3W, 1D, 0L)
        "away_form_win_rate":   [],
        "away_form_goal_avg":   [],
        "away_form_concede_avg":[],
        "away_form_pts5":       [],
    }

    def form_stats(q: deque):
        if not q:
            return 0.33, 1.0, 1.0, 3.0   # neutral defaults
        wins   = sum(1 for m in q if m["outcome"] == "Win")
        draws  = sum(1 for m in q if m["outcome"] == "Draw")
        gf     = sum(m["gf"] for m in q)
        ga     = sum(m["ga"] for m in q)
        n      = len(q)
        last5  = list(q)[-5:]
        pts5   = sum(3 if m["outcome"]=="Win" else 1 if m["outcome"]=="Draw" else 0
                     for m in last5)
        return wins/n, gf/n, ga/n, pts5

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]

        hw, hg, hc, hp = form_stats(history[h])
        aw, ag, ac, ap = form_stats(history[a])

        cols["home_form_win_rate"].append(hw)
        cols["home_form_goal_avg"].append(hg)
        cols["home_form_concede_avg"].append(hc)
        cols["home_form_pts5"].append(hp)
        cols["away_form_win_rate"].append(aw)
        cols["away_form_goal_avg"].append(ag)
        cols["away_form_concede_avg"].append(ac)
        cols["away_form_pts5"].append(ap)

        # Update history AFTER recording
        history[h].append({
            "outcome": row["outcome"],
            "gf": row["home_score"],
            "ga": row["away_score"]
        })
        away_outcome = {"Win":"Loss","Loss":"Win","Draw":"Draw"}[row["outcome"]]
        history[a].append({
            "outcome": away_outcome,
            "gf": row["away_score"],
            "ga": row["home_score"]
        })

    for col, vals in cols.items():
        df[col] = vals

    # Save final form snapshot for inference
    final_form = {}
    for team, q in history.items():
        stats = form_stats(q)
        final_form[team] = {
            "win_rate":    round(stats[0], 4),
            "goal_avg":    round(stats[1], 4),
            "concede_avg": round(stats[2], 4),
            "form_pts5":   round(stats[3], 4),
        }

    print(f" Form features computed for {len(final_form):,} teams")
    return df, final_form


# HEAD-TO-HEAD FEATURE

def compute_h2h(df: pd.DataFrame):
    # Pre-index all matches by pair for speed
    h2h_map = defaultdict(lambda: {"wins":0, "total":0})
    h2h_rates = []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        key  = (h, a)
        rec  = h2h_map[key]
        rate = rec["wins"] / rec["total"] if rec["total"] > 0 else 0.5
        h2h_rates.append(rate)

        # Update after recording
        if row["outcome"] == "Win":
            h2h_map[key]["wins"] += 1
        h2h_map[key]["total"] += 1
        # Also track reverse direction
        rev = (a, h)
        if row["outcome"] == "Loss":
            h2h_map[rev]["wins"] += 1
        h2h_map[rev]["total"] += 1

    df = df.copy()
    df["h2h_home_win_rate"] = h2h_rates
    print(" H2H features computed")
    return df

# TRAIN MODEL

FEATURES = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "home_form_win_rate",
    "home_form_goal_avg",
    "home_form_concede_avg",
    "home_form_pts5",
    "away_form_win_rate",
    "away_form_goal_avg",
    "away_form_concede_avg",
    "away_form_pts5",
    "h2h_home_win_rate",
    "is_neutral",
    "is_wc",
    "t_weight",
]


def train_model(df: pd.DataFrame):
    le = LabelEncoder()
    y  = le.fit_transform(df["outcome"])   # Draw=0, Loss=1, Win=2 (sorted)

    X = df[FEATURES].fillna(0)

    # ── Time-based split (more realistic than random) ──
    cutoff  = pd.Timestamp(EVAL_FROM)
    mask    = df["date"] < cutoff
    X_train, X_test = X[mask], X[~mask]
    y_train, y_test = y[mask], y[~mask]

    print(f"\n Train: {len(X_train):,} matches (before {EVAL_FROM})")
    print(f"   Test:  {len(X_test):,} matches  (from  {EVAL_FROM} onward)")

    # ── XGBoost ──
    model = XGBClassifier(
        n_estimators      = 600,
        max_depth         = 5,
        learning_rate     = 0.04,
        subsample         = 0.80,
        colsample_bytree  = 0.75,
        min_child_weight  = 5,
        gamma             = 0.1,
        reg_alpha         = 0.1,
        reg_lambda        = 1.5,
        objective         = "multi:softprob",
        num_class         = 3,
        eval_metric       = "mlogloss",
        use_label_encoder = False,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set        = [(X_test, y_test)],
        verbose         = False,
    )

    # ── Calibrate probabilities for better betting-style confidence scores ──
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_test, y_test)

    y_pred = calibrated.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n Model trained!")
    print(f"   Test Accuracy: {acc:.2%}")
    print(f"\nClassification Report (test set):")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_,
                                digits=3))

    # Feature importance
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x:-x[1])
    print(" Top features by importance:")
    for feat, score in imp[:8]:
        bar = "█" * int(score * 60)
        print(f"   {feat:<28} {score:.3f}  {bar}")

    return calibrated, le

# SAVE ALL ARTIFACTS

def save_artifacts(model, le, df, final_elo, final_form):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model,  os.path.join(MODEL_DIR, "xgb_model.pkl"))
    joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(final_elo,  os.path.join(MODEL_DIR, "team_elo.pkl"))
    joblib.dump(final_form, os.path.join(MODEL_DIR, "team_form.pkl"))
    print(f"\n Model  → model/xgb_model.pkl")

    # Teams list
    all_teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
    with open(os.path.join(MODEL_DIR, "teams.json"), "w") as f:
        json.dump(all_teams, f)
    print(f" Teams  → model/teams.json  ({len(all_teams):,} teams)")

    # Match history for H2H lookup in the app
    history = df[["date","home_team","away_team","home_score","away_score",
                  "outcome","tournament"]].copy()
    history["date"]  = history["date"].astype(str)
    history["Year"]  = history["date"].str[:4]
    # Rename to match app.py expectations
    history = history.rename(columns={
        "home_team":  "Home Team Name",
        "away_team":  "Away Team Name",
        "home_score": "Home Team Goals",
        "away_score": "Away Team Goals",
        "outcome":    "Outcome",
    })
    history.to_json(os.path.join(MODEL_DIR, "match_history.json"), orient="records")
    print(f" History → model/match_history.json  ({len(history):,} matches)")

    # Feature list (needed by app.py for inference)
    with open(os.path.join(MODEL_DIR, "features.json"), "w") as f:
        json.dump(FEATURES, f)
    print(f" Features → model/features.json")


# MAIN

if __name__ == "__main__":
    print("=" * 60)
    print("  GoalIQ — Enhanced Training Pipeline")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"\n Dataset not found at {DATA_PATH}")
        print("   Download from:")
        print("   kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017")
        print("   Place results.csv in the data/ folder, then run again.\n")
        exit(1)

    df              = load_data(DATA_PATH)
    df, final_elo   = compute_elo(df)
    df, final_form  = compute_form(df)
    df              = compute_h2h(df)
    model, le       = train_model(df)
    save_artifacts(model, le, df, final_elo, final_form)

    print("\n Training complete!")
    print("   Run `python app.py` to start the server.")
    print("\n REMINDER: Add a betting disclaimer to the app.")
    print("   Predictions are probabilistic estimates, not guarantees.\n")