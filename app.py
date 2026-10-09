from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
from datetime import datetime
from functools import wraps
import math
from ml_engine import model_ready, get_metrics, get_importance, predict_waste

app = Flask(__name__)
app.secret_key = "smartwaste-demo-secret-key"
DB = "smartwaste.db"

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ngos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            lat REAL DEFAULT 0,
            lon REAL DEFAULT 0,
            capacity INTEGER DEFAULT 50,
            required_food TEXT DEFAULT 'Any'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS donations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            donor_id INTEGER NOT NULL,
            food_name TEXT NOT NULL,
            food_type TEXT NOT NULL,
            quantity REAL NOT NULL,
            location TEXT NOT NULL,
            lat REAL DEFAULT 0,
            lon REAL DEFAULT 0,
            prepared_time TEXT NOT NULL,
            expiry_time TEXT NOT NULL,
            status TEXT DEFAULT 'Available',
            ngo_id INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY(donor_id) REFERENCES users(id),
            FOREIGN KEY(ngo_id) REFERENCES ngos(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            donation_id INTEGER NOT NULL,
            ngo_id INTEGER NOT NULL,
            score REAL NOT NULL,
            reason TEXT NOT NULL,
            status TEXT DEFAULT 'Pending',
            created_at TEXT NOT NULL,
            FOREIGN KEY(donation_id) REFERENCES donations(id),
            FOREIGN KEY(ngo_id) REFERENCES ngos(id)
        )
    """)

    # Demo users
    users = [
        ("Demo Donor", "donor@demo.com", "1234", "donor"),
        ("Demo NGO", "ngo@demo.com", "1234", "ngo"),
        ("Admin", "admin@demo.com", "admin", "admin")
    ]
    for u in users:
        cur.execute("INSERT OR IGNORE INTO users(name,email,password,role) VALUES(?,?,?,?)", u)

    # Demo NGOs
    ngo_data = [
        ("Hope Community Kitchen", "Coimbatore", 11.0168, 76.9558, 100, "Any"),
        ("Annai Food Care", "Peelamedu", 11.0300, 77.0000, 80, "Cooked Food"),
        ("Helping Hands Centre", "Gandhipuram", 11.0183, 76.9725, 120, "Rice"),
        ("Green Community Trust", "Singanallur", 11.0000, 77.0300, 70, "Vegetables")
    ]
    for n in ngo_data:
        cur.execute("""
            INSERT INTO ngos(name,location,lat,lon,capacity,required_food)
            SELECT ?,?,?,?,?,?
            WHERE NOT EXISTS (SELECT 1 FROM ngos WHERE name=?)
        """, (*n, n[0]))

    conn.commit()
    conn.close()

def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                flash("You do not have permission to access this page.", "danger")
                return redirect(url_for("home"))
            return f(*args, **kwargs)
        return wrapper
    return decorator

def distance_km(lat1, lon1, lat2, lon2):
    # Haversine distance
    if not all([lat1, lon1, lat2, lon2]):
        return 5.0
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def calculate_match(donation, ngo):
    """Explainable AI-style scoring for the prototype."""
    dist = distance_km(donation["lat"], donation["lon"], ngo["lat"], ngo["lon"])
    distance_score = max(0, 40 - dist * 5)

    try:
        expiry = datetime.fromisoformat(donation["expiry_time"])
        hours_left = max(0, (expiry - datetime.now()).total_seconds()/3600)
    except Exception:
        hours_left = 6

    urgency_score = max(0, 35 - hours_left * 4)
    requirement_score = 20 if ngo["required_food"].lower() in ("any", donation["food_type"].lower(), donation["food_name"].lower()) else 8
    capacity_score = min(5, donation["quantity"] / max(1, ngo["capacity"]) * 5)

    score = distance_score + urgency_score + requirement_score + capacity_score

    reasons = []
    if dist <= 3:
        reasons.append(f"nearby ({dist:.1f} km)")
    if hours_left <= 3:
        reasons.append("high expiry urgency")
    if requirement_score >= 20:
        reasons.append(f"food type matches {ngo['required_food']} requirement")
    if donation["quantity"] <= ngo["capacity"]:
        reasons.append("capacity is sufficient")

    reason = ", ".join(reasons) if reasons else "best overall score based on distance, urgency and capacity"
    return round(score, 2), round(dist, 2), round(hours_left, 1), reason

@app.route("/")
def home():
    conn = get_db()
    stats = {
        "food": conn.execute("SELECT COALESCE(SUM(quantity),0) FROM donations WHERE status IN ('Available','Accepted','Delivered')").fetchone()[0],
        "donations": conn.execute("SELECT COUNT(*) FROM donations").fetchone()[0],
        "delivered": conn.execute("SELECT COUNT(*) FROM donations WHERE status='Delivered'").fetchone()[0],
        "people": conn.execute("SELECT COALESCE(SUM(quantity * 4),0) FROM donations WHERE status='Delivered'").fetchone()[0]
    }
    conn.close()
    stats["co2"] = round(float(stats["food"]) * 2.5 / 1000, 2)
    return render_template("index.html", stats=stats)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip()
        password = request.form["password"]
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=? AND password=?", (email,password)).fetchone()
        conn.close()
        if user:
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
@login_required()
def dashboard():
    role = session["role"]
    if role == "donor":
        return redirect(url_for("donor_dashboard"))
    if role == "ngo":
        return redirect(url_for("ngo_dashboard"))
    return redirect(url_for("admin_dashboard"))

@app.route("/donor")
@login_required("donor")
def donor_dashboard():
    conn = get_db()
    donations = conn.execute("""
        SELECT d.*, n.name AS ngo_name
        FROM donations d LEFT JOIN ngos n ON d.ngo_id=n.id
        WHERE d.donor_id=? ORDER BY d.id DESC
    """, (session["user_id"],)).fetchall()
    conn.close()
    return render_template("donor.html", donations=donations)

@app.route("/donation/new", methods=["GET", "POST"])
@login_required("donor")
def new_donation():
    if request.method == "POST":
        try:
            food_name = request.form["food_name"]
            food_type = request.form["food_type"]
            quantity = float(request.form["quantity"])
            location = request.form["location"]
            lat = float(request.form.get("lat") or 11.0168)
            lon = float(request.form.get("lon") or 76.9558)
            prepared = request.form["prepared_time"]
            expiry = request.form["expiry_time"]
        except Exception:
            flash("Please enter valid donation details.", "danger")
            return render_template("new_donation.html")

        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO donations
            (donor_id,food_name,food_type,quantity,location,lat,lon,prepared_time,expiry_time,status,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,'Available',?)
        """, (session["user_id"],food_name,food_type,quantity,location,lat,lon,prepared,expiry,datetime.now().isoformat()))
        donation_id = cur.lastrowid

        donation = conn.execute("SELECT * FROM donations WHERE id=?", (donation_id,)).fetchone()
        ngos = conn.execute("SELECT * FROM ngos").fetchall()

        best = None
        for ngo in ngos:
            score, dist, hours, reason = calculate_match(donation, ngo)
            if best is None or score > best[0]:
                best = (score, ngo, dist, hours, reason)

        if best:
            score, ngo, dist, hours, reason = best
            cur.execute("INSERT INTO requests(donation_id,ngo_id,score,reason,created_at) VALUES(?,?,?,?,?)",
                        (donation_id, ngo["id"], score, reason, datetime.now().isoformat()))
            flash(f"Donation posted! AI recommends {ngo['name']} ({score} score).", "success")

        conn.commit()
        conn.close()
        return redirect(url_for("donor_dashboard"))
    return render_template("new_donation.html")

@app.route("/ngo")
@login_required("ngo")
def ngo_dashboard():
    conn = get_db()
    available = conn.execute("""
        SELECT d.*, u.name AS donor_name
        FROM donations d JOIN users u ON d.donor_id=u.id
        WHERE d.status='Available' ORDER BY d.id DESC
    """).fetchall()

    ngos = conn.execute("SELECT * FROM ngos").fetchall()
    my_ngo = ngos[0]  # demo NGO account represents first NGO

    scored = []
    for d in available:
        score, dist, hours, reason = calculate_match(d, my_ngo)
        scored.append({"donation": d, "score": score, "distance": dist, "hours": hours, "reason": reason})
    scored.sort(key=lambda x: x["score"], reverse=True)

    history = conn.execute("""
        SELECT d.*, n.name AS ngo_name
        FROM donations d LEFT JOIN ngos n ON d.ngo_id=n.id
        WHERE d.ngo_id=? ORDER BY d.id DESC
    """, (my_ngo["id"],)).fetchall()
    conn.close()
    return render_template("ngo.html", scored=scored, history=history, my_ngo=my_ngo)

@app.route("/ngo/accept/<int:donation_id>", methods=["POST"])
@login_required("ngo")
def accept_donation(donation_id):
    conn = get_db()
    my_ngo = conn.execute("SELECT * FROM ngos ORDER BY id LIMIT 1").fetchone()
    donation = conn.execute("SELECT * FROM donations WHERE id=? AND status='Available'", (donation_id,)).fetchone()
    if not donation:
        conn.close()
        flash("Donation is no longer available.", "danger")
        return redirect(url_for("ngo_dashboard"))

    conn.execute("UPDATE donations SET status='Accepted', ngo_id=? WHERE id=?", (my_ngo["id"], donation_id))
    conn.execute("UPDATE requests SET status='Accepted' WHERE donation_id=? AND ngo_id=?", (donation_id, my_ngo["id"]))
    conn.commit()
    conn.close()
    flash("Donation accepted. Pickup can now be arranged.", "success")
    return redirect(url_for("ngo_dashboard"))

@app.route("/ngo/deliver/<int:donation_id>", methods=["POST"])
@login_required("ngo")
def deliver_donation(donation_id):
    conn = get_db()
    conn.execute("UPDATE donations SET status='Delivered' WHERE id=? AND status='Accepted'", (donation_id,))
    conn.commit()
    conn.close()
    flash("Donation marked as delivered.", "success")
    return redirect(url_for("ngo_dashboard"))

@app.route("/admin")
@login_required("admin")
def admin_dashboard():
    conn = get_db()
    donations = conn.execute("""
        SELECT d.*, u.name AS donor_name, n.name AS ngo_name
        FROM donations d
        JOIN users u ON d.donor_id=u.id
        LEFT JOIN ngos n ON d.ngo_id=n.id
        ORDER BY d.id DESC
    """).fetchall()
    ngos = conn.execute("SELECT * FROM ngos").fetchall()
    total_food = conn.execute("SELECT COALESCE(SUM(quantity),0) FROM donations").fetchone()[0]
    delivered_food = conn.execute("SELECT COALESCE(SUM(quantity),0) FROM donations WHERE status='Delivered'").fetchone()[0]
    conn.close()

    return render_template("admin.html", donations=donations, ngos=ngos,
                           total_food=round(total_food,2),
                           delivered_food=round(delivered_food,2),
                           people=int(delivered_food*4))

@app.route("/api/recommend/<int:donation_id>")
@login_required()
def recommend(donation_id):
    conn = get_db()
    donation = conn.execute("SELECT * FROM donations WHERE id=?", (donation_id,)).fetchone()
    ngos = conn.execute("SELECT * FROM ngos").fetchall()
    conn.close()
    if not donation:
        return jsonify({"error":"Donation not found"}), 404

    results = []
    for ngo in ngos:
        score, dist, hours, reason = calculate_match(donation, ngo)
        results.append({
            "ngo": ngo["name"], "score": score, "distance_km": dist,
            "hours_left": hours, "reason": reason
        })
    return jsonify(sorted(results, key=lambda x: x["score"], reverse=True))


@app.route("/ml-lab", methods=["GET", "POST"])
def ml_dashboard():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            quantity = float(request.form.get("quantity", "400"))
            guests = int(request.form.get("guests", "300"))
            if quantity <= 0 or guests <= 0:
                raise ValueError("Food quantity and guest count must be greater than zero.")
            values = {
                "Type of Food": request.form.get("food_type", "Meat"),
                "Number of Guests": guests,
                "Event Type": request.form.get("event_type", "Corporate"),
                "Quantity of Food": quantity,
                "Storage Conditions": request.form.get("storage", "Refrigerated"),
                "Purchase History": request.form.get("purchase", "Regular"),
                "Seasonality": request.form.get("season", "All Seasons"),
                "Preparation Method": request.form.get("method", "Buffet"),
                "Geographical Location": request.form.get("location", "Urban"),
                "Pricing": request.form.get("pricing", "Moderate"),
            }
            prediction = predict_waste(values)
        except Exception as exc:
            error = str(exc)
    return render_template("ml_dashboard.html", ready=model_ready(), metrics=get_metrics(),
                           importance=get_importance(), prediction=prediction, error=error)

@app.route("/impact")
def impact():
    conn = get_db()
    food = conn.execute("SELECT COALESCE(SUM(quantity),0) FROM donations WHERE status='Delivered'").fetchone()[0]
    donations = conn.execute("SELECT COUNT(*) FROM donations WHERE status='Delivered'").fetchone()[0]
    conn.close()
    return render_template("impact.html", food=round(food,2), donations=donations,
                           people=int(food*4), co2=round(food*2.5/1000,2))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
