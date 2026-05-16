"""Flask application for the Fertilizer Recommendation System."""

from datetime import datetime, timedelta
from functools import wraps
from io import BytesIO
from pathlib import Path
import os
import pickle
import secrets
import sqlite3

import pandas as pd
import requests
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    send_file,
    url_for,
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from werkzeug.security import check_password_hash, generate_password_hash


# Core app paths keep the project portable for local and Render deployments.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
CROP_MODEL_PATH = BASE_DIR / "model" / "crop_model.pkl"
DATABASE_PATH = BASE_DIR / "database" / "fertilizer.db"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")


FERTILIZER_GUIDE = {
    "Urea": {
        "reason": "Urea is rich in nitrogen and supports vigorous leaf growth when nitrogen demand is high.",
        "tips": "Keep soil moisture balanced, avoid over-irrigation, and add organic matter after harvest.",
        "usage": "Apply in split doses near the root zone, then irrigate lightly to reduce nitrogen loss.",
    },
    "DAP": {
        "reason": "DAP supplies phosphorus and nitrogen, helping early root development and crop establishment.",
        "tips": "Maintain neutral soil pH where possible and use compost to improve phosphorus availability.",
        "usage": "Place near sowing rows before or during planting, avoiding direct seed contact.",
    },
    "10-26-26": {
        "reason": "This balanced NPK fertilizer is helpful where phosphorus and potassium are both needed.",
        "tips": "Test soil periodically and combine chemical fertilizer with farmyard manure for resilience.",
        "usage": "Broadcast evenly or apply as a basal dose according to local agronomy recommendations.",
    },
    "MOP": {
        "reason": "MOP provides potassium, which improves boll formation, water regulation, and crop strength.",
        "tips": "Improve drainage in heavy soils and keep salinity under control before repeated potash use.",
        "usage": "Apply as a basal or side-dress dose and irrigate after application for better uptake.",
    },
    "NPK": {
        "reason": "NPK supports balanced nutrition when nitrogen, phosphorus, and potassium are all moderately required.",
        "tips": "Use crop residue and compost to build soil structure and reduce nutrient leaching.",
        "usage": "Apply evenly around the crop root zone, following the recommended dose for the crop stage.",
    },
    "Compost": {
        "reason": "Compost improves soil biology and is suitable when chemical nutrient demand is lower.",
        "tips": "Add organic biomass regularly, mulch exposed soil, and avoid compacting wet fields.",
        "usage": "Mix well-decomposed compost into the topsoil before sowing or use around plants as mulch.",
    },
}


def load_model_artifact() -> dict:
    """Load the trained model, training it once if model.pkl is missing."""
    if not MODEL_PATH.exists():
        from model.train_model import train_model

        train_model()

    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def load_crop_model_artifact() -> dict:
    """Load the crop model, training it once if crop_model.pkl is missing."""
    if not CROP_MODEL_PATH.exists():
        from model.train_crop_model import train_crop_model

        train_crop_model()

    with CROP_MODEL_PATH.open("rb") as file:
        return pickle.load(file)


MODEL_ARTIFACT = load_model_artifact()
MODEL = MODEL_ARTIFACT["model"]
CROP_MODEL_ARTIFACT = load_crop_model_artifact()
CROP_MODEL = CROP_MODEL_ARTIFACT["model"]


def get_db_connection() -> sqlite3.Connection:
    """Open a SQLite connection with dictionary-style row access."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database() -> None:
    """Create the prediction history table if it does not already exist."""
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                reset_token TEXT,
                reset_expires_at TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                nitrogen REAL NOT NULL,
                phosphorus REAL NOT NULL,
                potassium REAL NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                moisture REAL NOT NULL,
                soil_type TEXT NOT NULL,
                crop_type TEXT NOT NULL,
                fertilizer_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                irrigation_level TEXT,
                irrigation_reason TEXT,
                irrigation_schedule TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS crop_recommendation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                nitrogen REAL NOT NULL,
                phosphorus REAL NOT NULL,
                potassium REAL NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                rainfall REAL NOT NULL,
                recommended_crop TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(prediction_history)").fetchall()
        }
        optional_columns = {
            "user_id": "INTEGER",
            "irrigation_level": "TEXT",
            "irrigation_reason": "TEXT",
            "irrigation_schedule": "TEXT",
        }
        for column_name, column_type in optional_columns.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE prediction_history ADD COLUMN {column_name} {column_type}"
                )


def get_current_user():
    """Return the logged-in user row, or None when the visitor is anonymous."""
    user_id = session.get("user_id")
    if not user_id:
        return None

    with get_db_connection() as connection:
        return connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def login_required(view):
    """Protect routes that should only be used by authenticated farmers."""
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please login to continue.", "warning")
            return redirect(url_for("login", next=request.full_path))
        return view(*args, **kwargs)

    return wrapped_view


def find_user_by_email(email: str):
    """Find a user row by normalized email address."""
    with get_db_connection() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE lower(email) = lower(?)",
            (email.strip(),),
        ).fetchone()


def recommend_irrigation(form_data: dict) -> dict:
    """Recommend irrigation level using moisture, weather, and crop water demand."""
    moisture = float(form_data["Moisture"])
    temperature = float(form_data["Temperature"])
    humidity = float(form_data["Humidity"])
    crop_type = str(form_data["Crop_Type"]).lower()

    # Crop demand adjusts the baseline score for water-loving or drought-tolerant crops.
    high_water_crops = {"rice", "sugarcane"}
    medium_water_crops = {"wheat", "maize", "cotton"}
    crop_adjustment = 16 if crop_type in high_water_crops else 8 if crop_type in medium_water_crops else 2

    dryness_score = 0
    dryness_score += max(0, 55 - moisture) * 1.45
    dryness_score += max(0, temperature - 28) * 2.2
    dryness_score += max(0, 55 - humidity) * 0.9
    dryness_score += crop_adjustment

    if moisture >= 72 and humidity >= 65:
        level = "Low"
        schedule = "Avoid immediate irrigation. Recheck soil moisture in 24 to 36 hours."
        reason = "Soil moisture and humidity are already high, so extra water may cause nutrient leaching or root stress."
    elif dryness_score >= 58:
        level = "High"
        schedule = "Irrigate within 6 to 12 hours using a deeper but controlled watering cycle."
        reason = "Low moisture, warmer weather, or crop water demand indicate strong irrigation need."
    elif dryness_score >= 34:
        level = "Medium"
        schedule = "Irrigate within 12 to 24 hours with moderate water and monitor field wetness."
        reason = "The field has moderate water demand based on moisture, weather, and crop type."
    else:
        level = "Low"
        schedule = "Use light irrigation only if leaves show stress; otherwise monitor the next day."
        reason = "Current moisture and weather conditions do not show urgent water stress."

    return {
        "level": level,
        "reason": reason,
        "schedule": schedule,
    }


def save_prediction(form_data: dict, fertilizer: str, confidence: float, irrigation: dict) -> int:
    """Store a prediction request and return the new database row id."""
    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO prediction_history (
                user_id, nitrogen, phosphorus, potassium, temperature, humidity, moisture,
                soil_type, crop_type, fertilizer_name, confidence,
                irrigation_level, irrigation_reason, irrigation_schedule, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.get("user_id"),
                form_data["Nitrogen"],
                form_data["Phosphorus"],
                form_data["Potassium"],
                form_data["Temperature"],
                form_data["Humidity"],
                form_data["Moisture"],
                form_data["Soil_Type"],
                form_data["Crop_Type"],
                fertilizer,
                confidence,
                irrigation["level"],
                irrigation["reason"],
                irrigation["schedule"],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        return int(cursor.lastrowid)


def parse_prediction_form() -> dict:
    """Validate and normalize prediction form values."""
    numeric_fields = [
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Temperature",
        "Humidity",
        "Moisture",
    ]
    parsed = {}

    for field in numeric_fields:
        value = request.form.get(field, "").strip()
        if value == "":
            raise ValueError(f"{field} is required.")
        parsed[field] = float(value)

    parsed["Soil_Type"] = request.form.get("Soil_Type", "").strip()
    parsed["Crop_Type"] = request.form.get("Crop_Type", "").strip()

    if not parsed["Soil_Type"] or not parsed["Crop_Type"]:
        raise ValueError("Please select soil type and crop type.")

    return parsed


def make_prediction(form_data: dict) -> tuple[str, float]:
    """Run the ML model and return fertilizer name plus confidence."""
    input_frame = pd.DataFrame([form_data], columns=MODEL_ARTIFACT["features"])
    fertilizer = MODEL.predict(input_frame)[0]

    if hasattr(MODEL, "predict_proba"):
        probabilities = MODEL.predict_proba(input_frame)[0]
        confidence = round(float(max(probabilities)) * 100, 2)
    else:
        confidence = 0.0

    return fertilizer, confidence


def parse_crop_form() -> dict:
    """Validate and normalize crop recommendation form values."""
    fields = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "Rainfall"]
    parsed = {}
    for field in fields:
        value = request.form.get(field, "").strip()
        if value == "":
            raise ValueError(f"{field} is required.")
        parsed[field] = float(value)
    return parsed


def make_crop_recommendation(form_data: dict) -> tuple[str, float]:
    """Run the crop ML model and return crop name plus confidence."""
    input_frame = pd.DataFrame([form_data], columns=CROP_MODEL_ARTIFACT["features"])
    crop = CROP_MODEL.predict(input_frame)[0]

    if hasattr(CROP_MODEL, "predict_proba"):
        probabilities = CROP_MODEL.predict_proba(input_frame)[0]
        confidence = round(float(max(probabilities)) * 100, 2)
    else:
        confidence = 0.0

    return crop, confidence


def explain_crop_recommendation(form_data: dict, crop: str) -> str:
    """Create a beginner-friendly reason for the crop recommendation."""
    return (
        f"{crop} matches the entered NPK balance with temperature around "
        f"{form_data['Temperature']} C, humidity near {form_data['Humidity']}%, "
        f"and rainfall of {form_data['Rainfall']} mm."
    )


def save_crop_recommendation(form_data: dict, crop: str, confidence: float, reason: str) -> int:
    """Store a crop recommendation request and return its history id."""
    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO crop_recommendation_history (
                user_id, nitrogen, phosphorus, potassium, temperature,
                humidity, rainfall, recommended_crop, confidence, reason, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.get("user_id"),
                form_data["Nitrogen"],
                form_data["Phosphorus"],
                form_data["Potassium"],
                form_data["Temperature"],
                form_data["Humidity"],
                form_data["Rainfall"],
                crop,
                confidence,
                reason,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        return int(cursor.lastrowid)


def fetch_openweather(city: str, api_key: str) -> dict:
    """Fetch current weather from OpenWeather when the user provides an API key."""
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": api_key, "units": "metric"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "temperature": payload["main"]["temp"],
        "humidity": payload["main"]["humidity"],
        "city": payload.get("name", city),
        "source": "OpenWeather",
    }


def choose_best_location(locations: list[dict], city: str) -> dict:
    """Choose the best geocoding result for a city or city, country search."""
    query_parts = [part.strip().lower() for part in city.split(",") if part.strip()]
    city_query = query_parts[0] if query_parts else city.lower()
    country_query = query_parts[1] if len(query_parts) > 1 else ""
    country_aliases = {
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "gb": "united kingdom",
        "us": "united states",
        "u.s.": "united states",
        "usa": "united states",
        "uae": "united arab emirates",
    }
    normalized_country_query = country_aliases.get(country_query, country_query)

    def score(location: dict) -> int:
        location_name = str(location.get("name", "")).lower()
        country_name = str(location.get("country", "")).lower()
        country_code = str(location.get("country_code", "")).lower()
        admin_name = str(location.get("admin1", "")).lower()
        population = int(location.get("population") or 0)

        points = 0
        if location_name == city_query:
            points += 80
        elif city_query in location_name:
            points += 35
        country_matches = {country_name, country_code, admin_name}
        if normalized_country_query and normalized_country_query in country_matches:
            points += 60
        if population:
            points += min(population // 100000, 25)
        return points

    return sorted(locations, key=score, reverse=True)[0]


def get_json_with_retry(url: str, params: dict) -> dict:
    """Request JSON twice so short provider/network hiccups do not break autofill."""
    last_error = None
    for _ in range(2):
        try:
            response = requests.get(url, params=params, timeout=12)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
    raise last_error


def fetch_openmeteo(city: str) -> dict:
    """Fetch current weather from Open-Meteo without requiring an API key."""
    search_name = city.split(",", 1)[0].strip()
    geo_payload = get_json_with_retry(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": search_name, "count": 10, "language": "en", "format": "json"},
    )
    locations = geo_payload.get("results", [])
    if not locations:
        raise ValueError("City not found. Try a nearby larger city.")

    location = choose_best_location(locations, city)
    weather_payload = get_json_with_retry(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,relative_humidity_2m",
        },
    )
    current = weather_payload.get("current", {})

    return {
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "city": location.get("name", city),
        "country": location.get("country", ""),
        "admin": location.get("admin1", ""),
        "source": "Open-Meteo",
    }


@app.context_processor
def inject_template_data() -> dict:
    """Share dropdown options and current year with all templates."""
    return {
        "soil_types": MODEL_ARTIFACT["soil_types"],
        "crop_types": MODEL_ARTIFACT["crop_types"],
        "year": datetime.now().year,
        "current_user": get_current_user(),
    }


@app.route("/")
def home():
    """Render the homepage."""
    return render_template("index.html", page_title="Smart AI Fertilizer Recommendation System")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Create a new user account."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("Name, email, and password are required.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        elif password != confirm_password:
            flash("Passwords do not match.", "danger")
        elif find_user_by_email(email):
            flash("An account already exists with this email.", "warning")
        else:
            with get_db_connection() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (name, email, password_hash, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        name,
                        email,
                        generate_password_hash(password),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
                session["user_id"] = int(cursor.lastrowid)
                session["user_name"] = name
            flash("Registration successful. Welcome to FertiAI.", "success")
            return redirect(url_for("predict"))

    return render_template("register.html", page_title="Register")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login an existing user."""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = find_user_by_email(email)

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            flash("Logged in successfully.", "success")
            next_page = request.args.get("next") or url_for("predict")
            return redirect(next_page)

        flash("Invalid email or password.", "danger")

    return render_template("login.html", page_title="Login")


@app.route("/logout")
def logout():
    """Clear the user session."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    """Create a local demo reset link for users who forgot their password."""
    reset_link = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = find_user_by_email(email)

        if user:
            token = secrets.token_urlsafe(32)
            expires_at = (datetime.now() + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
            with get_db_connection() as connection:
                connection.execute(
                    """
                    UPDATE users
                    SET reset_token = ?, reset_expires_at = ?
                    WHERE id = ?
                    """,
                    (token, expires_at, user["id"]),
                )
            reset_link = url_for("reset_password", token=token, _external=True)

        flash("If the email exists, a password reset link has been generated.", "info")

    return render_template("forgot_password.html", page_title="Forgot Password", reset_link=reset_link)


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token: str):
    """Reset a password using a valid reset token."""
    with get_db_connection() as connection:
        user = connection.execute(
            "SELECT * FROM users WHERE reset_token = ?",
            (token,),
        ).fetchone()

    if not user:
        flash("Invalid or expired reset link.", "danger")
        return redirect(url_for("forgot_password"))

    expires_at = datetime.strptime(user["reset_expires_at"], "%Y-%m-%d %H:%M:%S")
    if datetime.now() > expires_at:
        flash("This reset link has expired. Please request a new one.", "warning")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        elif password != confirm_password:
            flash("Passwords do not match.", "danger")
        else:
            with get_db_connection() as connection:
                connection.execute(
                    """
                    UPDATE users
                    SET password_hash = ?, reset_token = NULL, reset_expires_at = NULL
                    WHERE id = ?
                    """,
                    (generate_password_hash(password), user["id"]),
                )
            flash("Password reset successful. Please login.", "success")
            return redirect(url_for("login"))

    return render_template("reset_password.html", page_title="Reset Password")


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    """Render the form and process fertilizer predictions."""
    if request.method == "POST":
        try:
            form_data = parse_prediction_form()
            fertilizer, confidence = make_prediction(form_data)
            irrigation = recommend_irrigation(form_data)
            history_id = save_prediction(form_data, fertilizer, confidence, irrigation)
            flash("Recommendation generated successfully.", "success")
            return redirect(url_for("result", history_id=history_id))
        except ValueError as error:
            flash(str(error), "danger")
        except Exception as error:
            flash(f"Prediction failed: {error}", "danger")

    return render_template("predict.html", page_title="Predict Fertilizer")


@app.route("/crop-recommendation", methods=["GET", "POST"])
@login_required
def crop_recommendation():
    """Recommend the best crop using soil nutrients and weather inputs."""
    crop_result = None
    if request.method == "POST":
        try:
            form_data = parse_crop_form()
            crop, confidence = make_crop_recommendation(form_data)
            reason = explain_crop_recommendation(form_data, crop)
            history_id = save_crop_recommendation(form_data, crop, confidence, reason)
            crop_result = {
                "history_id": history_id,
                "crop": crop,
                "confidence": confidence,
                "reason": reason,
                "inputs": form_data,
            }
            flash("Crop recommendation generated successfully.", "success")
        except ValueError as error:
            flash(str(error), "danger")
        except Exception as error:
            flash(f"Crop recommendation failed: {error}", "danger")

    with get_db_connection() as connection:
        recent_crops = connection.execute(
            """
            SELECT * FROM crop_recommendation_history
            WHERE user_id = ? OR user_id IS NULL
            ORDER BY id DESC LIMIT 5
            """,
            (session.get("user_id"),),
        ).fetchall()

    return render_template(
        "crop_recommendation.html",
        page_title="Crop Recommendation",
        crop_result=crop_result,
        recent_crops=recent_crops,
        crop_model_accuracy=CROP_MODEL_ARTIFACT["accuracy"],
    )


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """Return JSON predictions for JavaScript or future mobile clients."""
    try:
        payload = request.get_json(force=True)
        form_data = {
            "Nitrogen": float(payload["Nitrogen"]),
            "Phosphorus": float(payload["Phosphorus"]),
            "Potassium": float(payload["Potassium"]),
            "Temperature": float(payload["Temperature"]),
            "Humidity": float(payload["Humidity"]),
            "Moisture": float(payload["Moisture"]),
            "Soil_Type": payload["Soil_Type"],
            "Crop_Type": payload["Crop_Type"],
        }
        fertilizer, confidence = make_prediction(form_data)
        irrigation = recommend_irrigation(form_data)
        history_id = save_prediction(form_data, fertilizer, confidence, irrigation)
        return jsonify(
            {
                "fertilizer": fertilizer,
                "confidence": confidence,
                "irrigation": irrigation,
                "history_id": history_id,
            }
        )
    except Exception as error:
        return jsonify({"error": str(error)}), 400


@app.route("/api/crop-recommendation", methods=["POST"])
@login_required
def api_crop_recommendation():
    """Return JSON crop recommendations for future clients."""
    try:
        payload = request.get_json(force=True)
        form_data = {
            "Nitrogen": float(payload["Nitrogen"]),
            "Phosphorus": float(payload["Phosphorus"]),
            "Potassium": float(payload["Potassium"]),
            "Temperature": float(payload["Temperature"]),
            "Humidity": float(payload["Humidity"]),
            "Rainfall": float(payload["Rainfall"]),
        }
        crop, confidence = make_crop_recommendation(form_data)
        reason = explain_crop_recommendation(form_data, crop)
        history_id = save_crop_recommendation(form_data, crop, confidence, reason)
        return jsonify(
            {
                "crop": crop,
                "confidence": confidence,
                "reason": reason,
                "history_id": history_id,
            }
        )
    except Exception as error:
        return jsonify({"error": str(error)}), 400


@app.route("/result/<int:history_id>")
@login_required
def result(history_id: int):
    """Show a saved recommendation with advice cards."""
    with get_db_connection() as connection:
        prediction = connection.execute(
            """
            SELECT * FROM prediction_history
            WHERE id = ? AND (user_id = ? OR user_id IS NULL)
            """,
            (history_id, session.get("user_id")),
        ).fetchone()

    if prediction is None:
        flash("Recommendation not found.", "warning")
        return redirect(url_for("predict"))

    guide = FERTILIZER_GUIDE.get(prediction["fertilizer_name"], FERTILIZER_GUIDE["NPK"])
    irrigation = {
        "level": prediction["irrigation_level"],
        "reason": prediction["irrigation_reason"],
        "schedule": prediction["irrigation_schedule"],
    }
    if not all(irrigation.values()):
        irrigation = recommend_irrigation(
            {
                "Moisture": prediction["moisture"],
                "Temperature": prediction["temperature"],
                "Humidity": prediction["humidity"],
                "Crop_Type": prediction["crop_type"],
            }
        )
    return render_template(
        "result.html",
        page_title="Recommendation Result",
        prediction=prediction,
        guide=guide,
        irrigation=irrigation,
    )


@app.route("/dashboard")
@login_required
def dashboard():
    """Display prediction statistics using Chart.js."""
    with get_db_connection() as connection:
        total = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM prediction_history
            WHERE user_id = ? OR user_id IS NULL
            """,
            (session.get("user_id"),),
        ).fetchone()["count"]
        fertilizer_rows = connection.execute(
            """
            SELECT fertilizer_name, COUNT(*) AS count
            FROM prediction_history
            WHERE user_id = ? OR user_id IS NULL
            GROUP BY fertilizer_name
            ORDER BY count DESC
            """,
            (session.get("user_id"),),
        ).fetchall()
        recent_rows = connection.execute(
            """
            SELECT * FROM prediction_history
            WHERE user_id = ? OR user_id IS NULL
            ORDER BY id DESC LIMIT 8
            """,
            (session.get("user_id"),),
        ).fetchall()

    return render_template(
        "dashboard.html",
        page_title="Dashboard",
        total=total,
        fertilizer_rows=fertilizer_rows,
        recent_rows=recent_rows,
    )


@app.route("/about")
def about():
    """Render project information."""
    return render_template("about.html", page_title="About")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    """Render the contact form and show a success message on submit."""
    if request.method == "POST":
        flash("Thanks for reaching out. We will contact you soon.", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html", page_title="Contact")


@app.route("/api/weather")
def weather():
    """Fetch temperature and humidity from OpenWeather or a no-key fallback."""
    city = request.args.get("city", "").strip()
    api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()

    if not city:
        return jsonify({"error": "City is required."}), 400

    try:
        weather_data = fetch_openweather(city, api_key) if api_key else fetch_openmeteo(city)
        return jsonify(weather_data)
    except Exception:
        return jsonify({"error": "Unable to fetch weather for this city. Check spelling or try another city."}), 400


@app.route("/download-report/<int:history_id>")
@login_required
def download_report(history_id: int):
    """Generate a PDF recommendation report for a saved prediction."""
    with get_db_connection() as connection:
        prediction = connection.execute(
            """
            SELECT * FROM prediction_history
            WHERE id = ? AND (user_id = ? OR user_id IS NULL)
            """,
            (history_id, session.get("user_id")),
        ).fetchone()

    if prediction is None:
        flash("Report not found.", "warning")
        return redirect(url_for("predict"))

    guide = FERTILIZER_GUIDE.get(prediction["fertilizer_name"], FERTILIZER_GUIDE["NPK"])
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, title="Fertilizer Recommendation Report")
    styles = getSampleStyleSheet()

    rows = [
        ["Nitrogen", prediction["nitrogen"]],
        ["Phosphorus", prediction["phosphorus"]],
        ["Potassium", prediction["potassium"]],
        ["Temperature", prediction["temperature"]],
        ["Humidity", prediction["humidity"]],
        ["Moisture", prediction["moisture"]],
        ["Soil Type", prediction["soil_type"]],
        ["Crop Type", prediction["crop_type"]],
        ["Recommended Fertilizer", prediction["fertilizer_name"]],
        ["Confidence", f"{prediction['confidence']}%"],
    ]
    irrigation = {
        "level": prediction["irrigation_level"],
        "reason": prediction["irrigation_reason"],
        "schedule": prediction["irrigation_schedule"],
    }
    if not all(irrigation.values()):
        irrigation = recommend_irrigation(
            {
                "Moisture": prediction["moisture"],
                "Temperature": prediction["temperature"],
                "Humidity": prediction["humidity"],
                "Crop_Type": prediction["crop_type"],
            }
        )
    rows.extend(
        [
            ["Irrigation Level", irrigation["level"]],
            ["Irrigation Schedule", irrigation["schedule"]],
        ]
    )

    table = Table(rows, colWidths=[170, 260])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f5e9")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#123524")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b7d7bd")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story = [
        Paragraph("Fertilizer Recommendation Report", styles["Title"]),
        Spacer(1, 14),
        table,
        Spacer(1, 18),
        Paragraph("Why Recommended", styles["Heading2"]),
        Paragraph(guide["reason"], styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Soil Improvement Tips", styles["Heading2"]),
        Paragraph(guide["tips"], styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Usage Instructions", styles["Heading2"]),
        Paragraph(guide["usage"], styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Smart Irrigation Recommendation", styles["Heading2"]),
        Paragraph(f"Level: {irrigation['level']}", styles["BodyText"]),
        Paragraph(irrigation["reason"], styles["BodyText"]),
        Paragraph(irrigation["schedule"], styles["BodyText"]),
    ]
    document.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"fertilizer_report_{history_id}.pdf",
        mimetype="application/pdf",
    )


init_database()


if __name__ == "__main__":
    app.run(debug=True)
