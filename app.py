import pandas as pd
import sqlite3
import re
from flask import Flask, render_template, request, redirect, session, jsonify
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------- LOAD MODEL ----------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- DATABASE ----------------
def get_db():
    return sqlite3.connect("database.db")

with get_db() as db:
    db.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)

# ---------------- LOAD DATASET ----------------
jobs_df = pd.read_csv("jobs.csv")
jobs_df.columns = jobs_df.columns.str.strip()
jobs_df = jobs_df.dropna(subset=["Job Title", "Job Description"])
jobs_df = jobs_df.drop_duplicates(subset=["Job Title"])

ROLE_DATASET = {
    row["Job Title"]: row["Job Description"]
    for _, row in jobs_df.iterrows()
}

# ---------------- AUTH ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            return "Missing email or password"

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE email=? AND password=?",
            (email, password)
        ).fetchone()

        if user:
            session["user"] = email
            return redirect("/dashboard")

        return "Invalid Credentials"

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        try:
            db = get_db()
            db.execute(
                "INSERT INTO users(email,password) VALUES (?,?)",
                (email, password)
            )
            db.commit()
            return redirect("/")
        except:
            return "User already exists"

    return render_template("signup.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html", roles=ROLE_DATASET.keys())


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


# ---------------- HELPER FUNCTIONS ----------------
def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {
        "the","and","for","with","that","this","from","are","was","were",
        "have","has","had","you","your","their","our","will","shall",
        "can","could","should","would","about","into","using","such",
        "also","other","more","than","each","within"
    }
    return set([w for w in words if w not in stopwords])


def generate_projects(selected_job, missing_skills):

    projects = set()

    if missing_skills:
        for skill in missing_skills[:3]:

            skill = skill.lower()

            if "tensorflow" in skill or "pytorch" in skill or "model" in skill:
                projects.add("Build an end-to-end ML pipeline with training, evaluation and deployment.")

            elif "react" in skill or "frontend" in skill or "ui" in skill:
                projects.add("Develop a modern responsive web app using React with dynamic state management.")

            elif "api" in skill or "backend" in skill or "flask" in skill:
                projects.add("Create a REST API with authentication and database integration.")

            elif "docker" in skill or "deployment" in skill:
                projects.add("Containerize an application using Docker and deploy it to cloud.")

            elif "sql" in skill or "database" in skill:
                projects.add("Design and implement a scalable database-backed application.")

            elif "nlp" in skill:
                projects.add("Build a real-world NLP project such as sentiment analysis or chatbot.")

            else:
                projects.add(
                    f"Build a hands-on applied project integrating {skill} with real-world business use cases."
                )

    else:
        role = selected_job.lower()

        if "machine learning" in role:
            projects.add("Build a production-grade ML system with model monitoring and CI/CD.")
        elif "frontend" in role:
            projects.add("Create a high-performance web application with optimized UX and animations.")
        elif "backend" in role:
            projects.add("Architect a scalable backend system with caching and load balancing.")
        elif "data" in role:
            projects.add("Develop an end-to-end data analytics pipeline with visualization dashboard.")
        else:
            projects.add(f"Build an advanced domain-specific project in {selected_job}.")

    return list(projects)


# ---------------- ANALYZE ROUTE ----------------
@app.route("/analyze", methods=["POST"])
def analyze():

    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    resume_file = request.files.get("resume")
    selected_job = request.form.get("job")

    if not resume_file or not selected_job:
        return jsonify({"error": "Missing input"}), 400

    if selected_job not in ROLE_DATASET:
        return jsonify({"error": "Invalid role"}), 400

    # Extract resume text
    reader = PdfReader(resume_file)
    resume_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            resume_text += text.lower()

    if not resume_text.strip():
        return jsonify({"error": "Could not extract text from PDF"}), 400

    # Compute semantic similarity
    resume_vector = model.encode([resume_text])
    job_vector = model.encode([ROLE_DATASET[selected_job]])

    similarity = cosine_similarity(resume_vector, job_vector)[0][0]
    score = float(((similarity + 1) / 2) * 100)
    score = round(score, 2)

    # Keyword gap analysis
    job_keywords = extract_keywords(ROLE_DATASET[selected_job])
    resume_keywords = extract_keywords(resume_text)
    missing_skills = list(job_keywords - resume_keywords)

    # Improvements based on score
    if score < 50:
        improvements = [
            "Add more role-specific technical skills.",
            "Include measurable achievements.",
            "Mention tools and frameworks clearly."
        ]
    elif score < 75:
        improvements = [
            "Enhance project descriptions.",
            "Align resume keywords with job description."
        ]
    else:
        improvements = [
            "Excellent match. Minor refinements needed."
        ]

    # Generate role-specific projects
    projects = generate_projects(selected_job, missing_skills)

    return jsonify({
        "score": score,
        "role": selected_job,
        "improvements": improvements,
        "projects": projects
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)