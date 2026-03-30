from flask import Flask, request, render_template
import joblib
import mysql.connector
from datetime import datetime

app = Flask(__name__)

model = joblib.load("model/alzheimer_model.pkl")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/alzheimer-risk", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        
        age = int(request.form["age"])
        family_history = int(request.form["family_history"])
        memory_loss = int(request.form["memory_loss"])
        confusion = int(request.form["confusion"])
        cognitive_score = int(request.form["cognitive_score"])
        physical_activity = int(request.form["physical_activity"])
        sleep_quality = int(request.form["sleep_quality"])

        data = [[
            age,
            family_history,
            memory_loss,
            confusion,
            cognitive_score,
            physical_activity,
            sleep_quality
        ]]

        
        pred = model.predict(data)[0]

        if pred == 0:
            risk = "High"
        elif pred == 1:
            risk = "Low"
        else:
            risk = "Medium"

        
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO screenings
            (age, family_history, memory_loss, confusion, cognitive_score,
             physical_activity, sleep_quality, risk, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            age,
            family_history,
            memory_loss,
            confusion,
            cognitive_score,
            physical_activity,
            sleep_quality,
            risk,
            datetime.now()
        ))

        conn.commit()
        cursor.close()
        conn.close()

        return render_template("index.html", result=True, risk=risk)

    return render_template("index.html", result=False)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)

