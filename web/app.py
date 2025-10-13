from flask import Flask, render_template, request, jsonify
import os
import subprocess
import json
import sys

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_SCRIPT = os.path.join(BASE_DIR, "main.py")


PYTHON_CMD = sys.executable  



@app.route("/")
def home():
    """Serve the front-end page."""
    return render_template("index.html")


@app.route("/crawl", methods=["POST"])
def crawl_site():
    url = request.form.get("url")
    max_pages = request.form.get("max_pages", "5")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    
    cmd = [PYTHON_CMD, MAIN_SCRIPT, "crawl", url, "--max-pages", str(max_pages)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout, stderr = result.stdout.strip(), result.stderr.strip()

    if result.returncode != 0:
        return jsonify({"error": stderr or "Crawl failed"}), 500

    try:
        return jsonify(json.loads(stdout))
    except:
        return jsonify({"raw": stdout or stderr})




@app.route("/index", methods=["POST"])
def index_site():
    """Run the index command via subprocess."""
    cmd = [PYTHON_CMD, MAIN_SCRIPT, "index"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    stdout, stderr = result.stdout.strip(), result.stderr.strip()
    if result.returncode != 0:
        return jsonify({"error": stderr or "Indexing failed"}), 500

    try:
        return jsonify(json.loads(stdout))
    except:
        return jsonify({"raw": stdout or stderr})


@app.route("/ask", methods=["POST"])
def ask_question():
    """Run the ask command via subprocess."""
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    cmd = [PYTHON_CMD, MAIN_SCRIPT, "ask", question]

    result = subprocess.run(cmd, capture_output=True, text=True)

    stdout, stderr = result.stdout.strip(), result.stderr.strip()
    if result.returncode != 0:
        return jsonify({"error": stderr or "Ask command failed"}), 500

    try:
        return jsonify(json.loads(stdout))
    except:
        return jsonify({"raw": stdout or stderr})


if __name__ == "__main__":
    
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
