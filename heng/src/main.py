"""
P96 - Predictive Road Safety & Risk Intelligence
FastAPI Scoring API - Sprint 3

Approach B (ML model) base score + live multipliers on top

POST /score  - score a GPS location
GET  /health - check model status
GET  /test   - run built-in test cases

Risk bands: 0-30 LOW | 31-60 MEDIUM | 61-80 HIGH | 81-100 CRITICAL
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==============================================================================
# PATHS
# ==============================================================================
_REPO        = Path(__file__).resolve().parent.parent
_MODEL_PATH  = _REPO / "output" / "model.pkl"
_ENC_PATH    = _REPO / "output" / "encoders.pkl"
_FEAT_PATH   = _REPO / "output" / "feature_names.txt"
_SCHOOL_CSV  = _REPO / "raw" / "school_zones_cleaned.csv"

MODEL_VERSION = "RF300-v8-AUC0.638"

# ==============================================================================
# CONSTANTS
# ==============================================================================
SPEED_RISK_MAP = {0:0, 40:1, 50:2, 60:3, 75:4, 80:4, 90:5, 100:6, 110:6}
DARKNESS_MAP   = {1:0.00, 2:0.30, 3:0.60, 4:0.85, 5:1.00, 6:0.70, 9:0.40}

_CITIES = {
    "Melbourne":   (-37.8136, 144.9631),
    "Geelong":     (-38.1499, 144.3617),
    "Ballarat":    (-37.5622, 143.8503),
    "Bendigo":     (-36.7570, 144.2794),
    "Shepparton":  (-36.3833, 145.4000),
    "Mildura":     (-34.1856, 142.1620),
    "Wodonga":     (-36.1218, 146.8882),
    "Sale":        (-38.1004, 147.0659),
    "Warrnambool": (-38.3838, 142.4825),
    "Traralgon":   (-38.1954, 146.5402),
}


# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================

class ScoreRequest(BaseModel):
    latitude:         float
    longitude:        float
    timestamp:        str   = Field(..., example="2024-06-15T14:30:00")
    actual_speed_kmh: float = Field(default=0.0, ge=0)
    posted_speed_kmh: float = Field(default=60.0, ge=0)
    no_of_vehicles:   int   = Field(default=1, ge=1)
    driver_id:        Optional[str] = None


class ScoreResponse(BaseModel):
    risk_score:          float
    risk_band:           str
    base_score:          float
    speed_multiplier:    float
    school_multiplier:   float
    weather_multiplier:  float
    nearest_city:        str
    driver_id:           Optional[str]
    timestamp:           str
    latitude:            float
    longitude:           float
    model_version:       str
    stubs_active:        list


# ==============================================================================
# HELPERS
# ==============================================================================

def _risk_band(score: float) -> str:
    if score <= 30:  return "LOW"
    if score <= 60:  return "MEDIUM"
    if score <= 80:  return "HIGH"
    return "CRITICAL"


def _nearest_city(lat: float, lon: float) -> str:
    best, best_d = "Unknown", math.inf
    for city, (clat, clon) in _CITIES.items():
        dlat = math.radians(clat - lat)
        dlon = math.radians(clon - lon)
        a = (math.sin(dlat/2)**2
             + math.cos(math.radians(lat))
             * math.cos(math.radians(clat))
             * math.sin(dlon/2)**2)
        d = 2 * 6_371_000 * math.asin(math.sqrt(a))
        if d < best_d:
            best, best_d = city, d
    return best


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def _safe_encode(encoders: dict, col: str, value) -> int:
    if col not in encoders:
        return 0
    le = encoders[col]
    val_str = str(value)
    if val_str in le.classes_:
        return int(le.transform([val_str])[0])
    return 0


def _nearest_school_dist(lat, lon, school_lats, school_lons) -> float:
    dists = [
        _haversine_m(lat, lon, slat, slon)
        for slat, slon in zip(school_lats, school_lons)
    ]
    return float(min(dists))


# ==============================================================================
# MULTIPLIERS - applied AFTER model base score
# ==============================================================================

def _speed_multiplier(actual_kmh: float, posted_kmh: float) -> float:
    """Speed deviation: 1.0 to 1.5"""
    if posted_kmh <= 0:
        return 1.0
    deviation = max(0.0, actual_kmh - posted_kmh)
    if deviation == 0:    return 1.0
    elif deviation <= 10: return 1.1
    elif deviation <= 20: return 1.2
    elif deviation <= 30: return 1.35
    else:                 return 1.5


def _school_zone_multiplier(dist_m: float, ts: pd.Timestamp) -> float:
    """
    Ben's fuzzy school zone curve:
    - 0-200m AND active hours: flat 1.6
    - 200-600m AND active hours: taper 1.6 to 1.0
    - Outside active hours OR >600m: 1.0

    Active: 7:45-8:30am and 3:00-3:30pm Mon-Fri
    """
    hour   = ts.hour
    minute = ts.minute
    dow    = ts.dayofweek  # 0=Mon, 6=Sun

    is_weekday = dow < 5
    morning    = (hour == 7 and minute >= 45) or (hour == 8 and minute <= 30)
    afternoon  = (hour == 15 and minute <= 30)
    is_active  = is_weekday and (morning or afternoon)

    if not is_active:
        return 1.0
    if dist_m <= 200:
        return 1.6
    elif dist_m <= 600:
        return 1.6 - (dist_m - 200) / 400 * 0.6
    else:
        return 1.0


def _weather_multiplier(lat: float, lon: float, ts: pd.Timestamp) -> float:
    """
    STUB - Van Sung will replace this with live Open-Meteo API call.

    Should fetch: precipitation, wind_gusts, temperature
    Then calculate multiplier based on how bad conditions are.

    For now returns 1.0 (no effect).
    """
    # Van Sung: replace this entire function with:
    # 1. Call Open-Meteo API with lat, lon, ts
    # 2. Get precipitation, wind_gusts, temperature
    # 3. Calculate multiplier:
    #    precip > 5mm -> 1.3, precip > 1mm -> 1.15
    #    wind > 60kmh -> 1.2, wind > 30kmh -> 1.1
    #    temp < 2C -> 1.1 (ice risk)
    #    Combine: mult = precip_mult * wind_mult * temp_mult
    #    Cap at 1.6
    return 1.0


# ==============================================================================
# FEATURE BUILDER - must match feature_names.txt exactly
# ==============================================================================

def _build_features(
    req: ScoreRequest,
    ts: pd.Timestamp,
    nearest_school_dist_m: float,
    encoders: dict,
    feature_names: list,
) -> pd.DataFrame:
    """
    Build the 22-feature vector matching model training.
    Returns a DataFrame so sklearn gets feature names (no warnings).
    """
    hour = ts.hour

    # VicRoads DAY_OF_WEEK: 1=Sun, 2=Mon ... 7=Sat
    pandas_dow   = ts.dayofweek       # 0=Mon, 6=Sun
    vicroads_dow = (pandas_dow + 2) % 7 + 1

    # Speed zone from posted speed
    speed_zone = int(req.posted_speed_kmh) if req.posted_speed_kmh > 0 else 60
    if speed_zone in [777, 888]:
        speed_zone = 60

    # Road context from speed zone
    if speed_zone >= 100:
        road_class    = "Freeway_Highway"
        road_geometry = "Not at intersection"
        node_type     = "N"
        deg_urban     = "MELB_URBAN"
    elif speed_zone >= 80:
        road_class    = "Main_Road"
        road_geometry = "Not at intersection"
        node_type     = "N"
        deg_urban     = "MELB_URBAN"
    else:
        road_class    = "Local_Road"
        road_geometry = "T intersection"
        node_type     = "I"
        deg_urban     = "MELB_URBAN"

    # Light condition from hour
    light_condition = 1 if 6 <= hour <= 19 else 3

    # Calendar flags from timestamp
    month = ts.month
    day   = ts.day
    # Daylight saving: roughly first Sunday Oct to first Sunday Apr
    is_daylight_saving = 1 if (month >= 10 or month <= 3) else 0
    # Public holiday / school holiday: default 0 (would need calendar lookup)
    is_public_holiday = 0
    is_school_holiday = 0

    raw = {
        "is_weekend":            1 if vicroads_dow in [1, 7] else 0,
        "is_peak_hour":          1 if hour in [7, 8, 9, 16, 17, 18] else 0,
        "hour_sin":              math.sin(2 * math.pi * hour / 24),
        "hour_cos":              math.cos(2 * math.pi * hour / 24),
        "day_sin":               math.sin(2 * math.pi * vicroads_dow / 7),
        "day_cos":               math.cos(2 * math.pi * vicroads_dow / 7),
        "speed_risk":            SPEED_RISK_MAP.get(speed_zone, 3),
        "darkness_score":        DARKNESS_MAP.get(light_condition, 0.40),
        "ROAD_GEOMETRY_DESC":    _safe_encode(encoders, "ROAD_GEOMETRY_DESC", road_geometry),
        "DISTANCE_LOCATION":     0.0,
        "NODE_TYPE":             _safe_encode(encoders, "NODE_TYPE", node_type),
        "road_class":            _safe_encode(encoders, "road_class", road_class),
        "wet_road":              0,
        "LGA_NAME":              _safe_encode(encoders, "LGA_NAME", "MELBOURNE"),
        "DEG_URBAN_NAME":        _safe_encode(encoders, "DEG_URBAN_NAME", deg_urban),
        "NO_OF_VEHICLES":        req.no_of_vehicles,
        "aadt_volume":           0,       # STUB - Mehak wires in live lookup
        "crash_rate":            0.0,     # STUB - Mehak wires in live lookup
        "nearest_school_dist_m": nearest_school_dist_m,
        "is_public_holiday":     is_public_holiday,
        "is_school_holiday":     is_school_holiday,
        "is_daylight_saving":    is_daylight_saving,
    }

    return pd.DataFrame([[raw[f] for f in feature_names]], columns=feature_names)


# ==============================================================================
# STARTUP
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[startup] Loading ML model...")
    app.state.model         = joblib.load(_MODEL_PATH)
    app.state.encoders      = joblib.load(_ENC_PATH)
    app.state.feature_names = _FEAT_PATH.read_text(encoding="utf-8").splitlines()
    print(f"[startup] Model: {MODEL_VERSION}")
    print(f"[startup] Features: {len(app.state.feature_names)}")

    print("[startup] Loading school zones...")
    school_df = pd.read_csv(_SCHOOL_CSV, low_memory=False)
    app.state.school_lats = school_df["CENTROID_LAT"].values.astype(float)
    app.state.school_lons = school_df["CENTROID_LON"].values.astype(float)
    print(f"[startup] Schools: {len(school_df):,} centroids")

    print("[startup] Ready.\n")
    yield
    print("[shutdown] Bye.")


# ==============================================================================
# APP
# ==============================================================================

app = FastAPI(
    title       = "P96 Risk Scoring API",
    description = "Approach B + live multipliers",
    version     = "3.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ==============================================================================
# SCORING
# ==============================================================================

def _compute_score(req: ScoreRequest, state) -> ScoreResponse:
    ts = pd.Timestamp(req.timestamp)

    # School distance
    nearest_school = _nearest_school_dist(
        req.latitude, req.longitude,
        state.school_lats, state.school_lons,
    )

    # Build features
    features = _build_features(
        req, ts, nearest_school,
        state.encoders, state.feature_names,
    )

    # ML base score
    base_prob  = float(state.model.predict_proba(features)[0][1])
    base_score = round(base_prob * 100, 2)

    # Live multipliers
    speed_mult   = _speed_multiplier(req.actual_speed_kmh, req.posted_speed_kmh)
    school_mult  = _school_zone_multiplier(nearest_school, ts)
    weather_mult = _weather_multiplier(req.latitude, req.longitude, ts)

    # Combined final score
    final = min(100.0, base_score * speed_mult * school_mult * weather_mult)
    final = round(final, 2)

    # Track which stubs are active
    stubs = []
    if weather_mult == 1.0:
        stubs.append("weather (Van Sung)")
    stubs.append("aadt_volume (Mehak)")
    stubs.append("crash_rate (Mehak)")
    stubs.append("LGA_NAME (Mehak)")

    return ScoreResponse(
        risk_score         = final,
        risk_band          = _risk_band(final),
        base_score         = base_score,
        speed_multiplier   = speed_mult,
        school_multiplier  = round(school_mult, 3),
        weather_multiplier = weather_mult,
        nearest_city       = _nearest_city(req.latitude, req.longitude),
        driver_id          = req.driver_id,
        timestamp          = req.timestamp,
        latitude           = req.latitude,
        longitude          = req.longitude,
        model_version      = MODEL_VERSION,
        stubs_active       = stubs,
    )


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/health")
async def health(request: Request):
    return {
        "status":        "ok",
        "model_version": MODEL_VERSION,
        "features":      len(request.app.state.feature_names),
        "feature_list":  request.app.state.feature_names,
        "stubs":         ["weather", "aadt_volume", "crash_rate", "LGA_NAME"],
        "approach":      "B (ML base) + live multipliers (speed, school, weather)",
    }


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest, request: Request):
    return _compute_score(req, request.app.state)


@app.get("/test")
async def test(request: Request):
    cases = [
        {
            "label": "Test 1 - Melbourne CBD normal day 3pm 60kmh",
            "req": ScoreRequest(
                latitude=-37.8136, longitude=144.9631,
                timestamp="2024-06-15T15:00:00",
                actual_speed_kmh=60, posted_speed_kmh=60,
                driver_id="TEST-001",
            ),
        },
        {
            "label": "Test 2 - Monash Freeway speeding 25 over at 8am",
            "req": ScoreRequest(
                latitude=-37.8776, longitude=145.1543,
                timestamp="2024-04-25T08:00:00",
                actual_speed_kmh=125, posted_speed_kmh=100,
                driver_id="TEST-002",
            ),
        },
        {
            "label": "Test 3 - School zone pickup time 3:10pm 40kmh",
            "req": ScoreRequest(
                latitude=-37.8136, longitude=144.9631,
                timestamp="2024-05-10T15:10:00",
                actual_speed_kmh=40, posted_speed_kmh=40,
                driver_id="TEST-003",
            ),
        },
        {
            "label": "Test 4 - Rural highway night 100kmh",
            "req": ScoreRequest(
                latitude=-36.5000, longitude=144.5000,
                timestamp="2024-07-20T23:00:00",
                actual_speed_kmh=100, posted_speed_kmh=100,
                driver_id="TEST-004",
            ),
        },
        {
            "label": "Test 5 - Dangerous: speeding 50 over near school",
            "req": ScoreRequest(
                latitude=-37.8136, longitude=144.9631,
                timestamp="2024-05-10T15:10:00",
                actual_speed_kmh=110, posted_speed_kmh=60,
                driver_id="TEST-005",
            ),
        },
    ]
    results = []
    for tc in cases:
        scored = _compute_score(tc["req"], request.app.state)
        results.append({
            "label":  tc["label"],
            "score":  scored.risk_score,
            "band":   scored.risk_band,
            "base":   scored.base_score,
            "speed_mult":   scored.speed_multiplier,
            "school_mult":  scored.school_multiplier,
            "weather_mult": scored.weather_multiplier,
        })
    return {"tests": results, "model_version": MODEL_VERSION}


# ==============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")