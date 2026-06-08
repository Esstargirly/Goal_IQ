from flask import Flask, render_template, request, jsonify
from datetime import date
import joblib, json, numpy as np, os, requests

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# ── OFFICIAL 2026 WORLD CUP TEAMS (all 48) ──
WC2026_TEAMS = sorted([
    "Algeria", "Angola", "Argentina", "Australia", "Austria",
    "Belgium", "Bosnia and Herzegovina", "Brazil",
    "Canada", "Cape Verde", "Colombia", "Congo DR", "Curacao", "Czechia",
    "Denmark", "Ecuador", "Egypt", "England",
    "France", "Germany", "Haiti", "Iran", "Iraq", "Ivory Coast",
    "Japan", "Jordan", "Mexico", "Morocco",
    "Netherlands", "New Zealand", "Norway",
    "Panama", "Paraguay", "Portugal", "Qatar",
    "Saudi Arabia", "Scotland", "Senegal", "South Africa",
    "South Korea", "Spain", "Switzerland",
    "Tunisia", "Turkiye", "Ukraine", "United States",
    "Uruguay", "Uzbekistan",
])

# 2026 SCHEDULE (fallback if live API unavailable)
WC2026_SCHEDULE = [
    {"date":"2026-06-11","home":"Mexico","away":"South Africa","time":"19:00","venue":"Estadio Azteca","group":"Group A"},
    {"date":"2026-06-11","home":"South Korea","away":"Czechia","time":"22:00","venue":"Estadio Akron","group":"Group A"},
    {"date":"2026-06-12","home":"Canada","away":"Bosnia and Herzegovina","time":"15:00","venue":"BMO Field","group":"Group B"},
    {"date":"2026-06-12","home":"United States","away":"Paraguay","time":"21:00","venue":"SoFi Stadium","group":"Group D"},
    {"date":"2026-06-13","home":"Qatar","away":"Switzerland","time":"15:00","venue":"Levi's Stadium","group":"Group B"},
    {"date":"2026-06-13","home":"Brazil","away":"Morocco","time":"18:00","venue":"MetLife Stadium","group":"Group C"},
    {"date":"2026-06-14","home":"Scotland","away":"Haiti","time":"15:00","venue":"AT&T Stadium","group":"Group C"},
    {"date":"2026-06-14","home":"Germany","away":"Curacao","time":"18:00","venue":"Arrowhead Stadium","group":"Group E"},
    {"date":"2026-06-14","home":"Australia","away":"Turkiye","time":"21:00","venue":"NRG Stadium","group":"Group D"},
    {"date":"2026-06-15","home":"Spain","away":"Cape Verde","time":"12:00","venue":"Mercedes-Benz Stadium","group":"Group H"},
    {"date":"2026-06-15","home":"Netherlands","away":"Ukraine","time":"18:00","venue":"Lumen Field","group":"Group F"},
    {"date":"2026-06-15","home":"Belgium","away":"New Zealand","time":"21:00","venue":"Lincoln Financial Field","group":"Group G"},
    {"date":"2026-06-16","home":"France","away":"Senegal","time":"15:00","venue":"MetLife Stadium","group":"Group I"},
    {"date":"2026-06-16","home":"Argentina","away":"Jordan","time":"18:00","venue":"Hard Rock Stadium","group":"Group J"},
    {"date":"2026-06-16","home":"Ecuador","away":"Ivory Coast","time":"21:00","venue":"BC Place","group":"Group E"},
    {"date":"2026-06-17","home":"Japan","away":"Tunisia","time":"12:00","venue":"AT&T Stadium","group":"Group F"},
    {"date":"2026-06-17","home":"Iran","away":"Egypt","time":"15:00","venue":"Arrowhead Stadium","group":"Group G"},
    {"date":"2026-06-17","home":"Uruguay","away":"Saudi Arabia","time":"18:00","venue":"SoFi Stadium","group":"Group H"},
    {"date":"2026-06-17","home":"Norway","away":"Iraq","time":"21:00","venue":"Levi's Stadium","group":"Group I"},
    {"date":"2026-06-18","home":"Czechia","away":"South Africa","time":"12:00","venue":"Mercedes-Benz Stadium","group":"Group A"},
    {"date":"2026-06-18","home":"Portugal","away":"Angola","time":"18:00","venue":"Estadio Azteca","group":"Group K"},
    {"date":"2026-06-19","home":"Mexico","away":"Denmark","time":"19:00","venue":"Estadio Akron","group":"Group A"},
    {"date":"2026-06-20","home":"United States","away":"Australia","time":"18:00","venue":"AT&T Stadium","group":"Group D"},
    {"date":"2026-06-20","home":"Germany","away":"Ecuador","time":"21:00","venue":"Hard Rock Stadium","group":"Group E"},
    {"date":"2026-06-21","home":"Spain","away":"Saudi Arabia","time":"12:00","venue":"Mercedes-Benz Stadium","group":"Group H"},
    {"date":"2026-06-21","home":"Brazil","away":"Scotland","time":"18:00","venue":"AT&T Stadium","group":"Group C"},
    {"date":"2026-06-22","home":"Norway","away":"Senegal","time":"15:00","venue":"Lincoln Financial Field","group":"Group I"},
    {"date":"2026-06-22","home":"Belgium","away":"Iran","time":"18:00","venue":"Arrowhead Stadium","group":"Group G"},
    {"date":"2026-06-24","home":"Morocco","away":"Haiti","time":"18:00","venue":"Mercedes-Benz Stadium","group":"Group C"},
    {"date":"2026-06-24","home":"Ecuador","away":"Germany","time":"21:00","venue":"Estadio Azteca","group":"Group E"},
    {"date":"2026-06-25","home":"France","away":"Norway","time":"18:00","venue":"MetLife Stadium","group":"Group I"},
    {"date":"2026-06-27","home":"Congo DR","away":"Uzbekistan","time":"19:30","venue":"Mercedes-Benz Stadium","group":"Group L"},
    {"date":"2026-06-27","home":"Panama","away":"Algeria","time":"22:00","venue":"NRG Stadium","group":"Group K"},
]

# ── Load model artifacts ──
MODEL_DIR = "model"
try:
    model      = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    le         = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    team_elo   = joblib.load(os.path.join(MODEL_DIR, "team_elo.pkl"))
    team_form  = joblib.load(os.path.join(MODEL_DIR, "team_form.pkl"))
    with open(os.path.join(MODEL_DIR, "match_history.json")) as f: MATCH_HISTORY = json.load(f)
    print(f" Model loaded  |  {len(WC2026_TEAMS)} WC 2026 teams in dropdown")
except FileNotFoundError:
    print("  Model not found — run train.py first")
    model = le = team_elo = team_form = None
    MATCH_HISTORY = []

ELO_DEFAULT = 1500

# ── Feature builder ──
def build_features(home, away):
    h_elo  = team_elo.get(home, ELO_DEFAULT) if team_elo else ELO_DEFAULT
    a_elo  = team_elo.get(away, ELO_DEFAULT) if team_elo else ELO_DEFAULT
    h_form = team_form.get(home, {}) if team_form else {}
    a_form = team_form.get(away, {}) if team_form else {}
    return np.array([[
        h_elo - a_elo,
        h_elo,
        a_elo,
        h_form.get("win_rate",    0.33),
        h_form.get("goal_avg",    1.0),
        h_form.get("concede_avg", 1.0),
        h_form.get("form_pts5",   3.0),
        a_form.get("win_rate",    0.33),
        a_form.get("goal_avg",    1.0),
        a_form.get("concede_avg", 1.0),
        a_form.get("form_pts5",   3.0),
        0.5, 1, 1, 1.0,
    ]])

# LIVE SCHEDULE FETCH
FOOTBALL_API_KEY = os.environ.get("FOOTBALL_API_KEY", "")
WC2026_COMPETITION_ID = 2000 

def fetch_live_matches(target_date: str):
    if not FOOTBALL_API_KEY:
        return None  # No key — use fallback
    try:
        url = f"https://api.football-data.org/v4/competitions/{WC2026_COMPETITION_ID}/matches"
        headers = {"X-Auth-Token": FOOTBALL_API_KEY}
        params  = {"dateFrom": target_date, "dateTo": target_date}
        resp    = requests.get(url, headers=headers, params=params, timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json()
        matches = []
        for m in data.get("matches", []):
            matches.append({
                "date":  target_date,
                "home":  m["homeTeam"]["name"],
                "away":  m["awayTeam"]["name"],
                "time":  m["utcDate"][11:16],
                "venue": m.get("venue", "—"),
                "group": m.get("group", "Group Stage"),
                "status": m["status"],
                "home_score": m["score"]["fullTime"]["home"],
                "away_score": m["score"]["fullTime"]["away"],
            })
        return matches if matches else None
    except Exception:
        return None

# Routes
@app.route("/")
def index():
    return render_template("index.html", teams=WC2026_TEAMS)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    data      = request.get_json()
    home_team = data.get("home_team","").strip()
    away_team = data.get("away_team","").strip()

    if not home_team or not away_team:
        return jsonify({"error": "Both teams are required."}), 400
    if home_team == away_team:
        return jsonify({"error": "Please select two different teams."}), 400

    X       = build_features(home_team, away_team)
    proba   = model.predict_proba(X)[0]
    idx     = int(np.argmax(proba))
    outcome = le.inverse_transform([idx])[0]

    class_proba = {
        le.inverse_transform([i])[0]: round(float(p)*100, 1)
        for i, p in enumerate(proba)
    }

    h_elo  = team_elo.get(home_team, ELO_DEFAULT) if team_elo else ELO_DEFAULT
    a_elo  = team_elo.get(away_team, ELO_DEFAULT) if team_elo else ELO_DEFAULT
    h_form = team_form.get(home_team, {}) if team_form else {}
    a_form = team_form.get(away_team, {}) if team_form else {}

    return jsonify({
        "home_team":     home_team,
        "away_team":     away_team,
        "prediction":    outcome,
        "confidence":    round(float(proba[idx])*100, 1),
        "probabilities": class_proba,
        "home_stats": {
            "elo":         round(h_elo),
            "win_rate":    round(h_form.get("win_rate",0)*100, 1),
            "goal_avg":    round(h_form.get("goal_avg",0), 2),
            "concede_avg": round(h_form.get("concede_avg",0), 2),
        },
        "away_stats": {
            "elo":         round(a_elo),
            "win_rate":    round(a_form.get("win_rate",0)*100, 1),
            "goal_avg":    round(a_form.get("goal_avg",0), 2),
            "concede_avg": round(a_form.get("concede_avg",0), 2),
        }
    })

@app.route("/head-to-head")
def head_to_head():
    home = request.args.get("home","").strip()
    away = request.args.get("away","").strip()
    if not home or not away:
        return jsonify({"error":"Both teams required"}), 400

    matches = [
        m for m in MATCH_HISTORY
        if (m["Home Team Name"]==home and m["Away Team Name"]==away) or
           (m["Home Team Name"]==away and m["Away Team Name"]==home)
    ]
    matches = sorted(matches, key=lambda x: x.get("Year","0"), reverse=True)
    home_wins = sum(1 for m in matches if
        (m["Home Team Name"]==home and m["Outcome"]=="Win") or
        (m["Away Team Name"]==home and m["Outcome"]=="Loss"))
    away_wins = sum(1 for m in matches if
        (m["Home Team Name"]==away and m["Outcome"]=="Win") or
        (m["Away Team Name"]==away and m["Outcome"]=="Loss"))
    draws = sum(1 for m in matches if m["Outcome"]=="Draw")

    return jsonify({
        "home_team":home,"away_team":away,
        "total":len(matches),
        "home_wins":home_wins,"away_wins":away_wins,"draws":draws,
        "matches":matches[:10]
    })

@app.route("/schedule/today")
def today_schedule():
    today = date.today().isoformat()

    # 1. Try live API first
    live = fetch_live_matches(today)
    if live:
        return jsonify({"type":"today","date":today,"matches":live,"source":"live"})

    # 2. Fall back to hardcoded schedule
    today_m = [m for m in WC2026_SCHEDULE if m["date"]==today]
    if today_m:
        return jsonify({"type":"today","date":today,"matches":today_m,"source":"schedule"})

    future = [m for m in WC2026_SCHEDULE if m["date"]>today]
    if future:
        nd = min(m["date"] for m in future)
        return jsonify({"type":"upcoming","date":nd,
                        "matches":[m for m in WC2026_SCHEDULE if m["date"]==nd],
                        "source":"schedule"})

    past = [m for m in WC2026_SCHEDULE if m["date"]<=today]
    if past:
        ld = max(m["date"] for m in past)
        return jsonify({"type":"past","date":ld,
                        "matches":[m for m in WC2026_SCHEDULE if m["date"]==ld],
                        "source":"schedule"})

    return jsonify({"type":"none","matches":[]})

@app.route("/teams")
def teams():
    return jsonify({"teams": WC2026_TEAMS})

if __name__ == "__main__":
    app.run(debug=True, port=5000)