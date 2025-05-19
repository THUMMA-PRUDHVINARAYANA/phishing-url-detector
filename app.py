from flask import Flask, request, render_template
import joblib

# Load trained model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get URL from the form
    url = request.form.get("url")

    if not url:
        return render_template("index.html", error="Please enter a URL")

    # Vectorize the input URL
    url_vectorized = vectorizer.transform([url])

    # Make prediction
    prediction = model.predict(url_vectorized)

    # Prepare result text
    result = "Phishing Website 🔴" if prediction[0] == 1 else "Legitimate Website 🟢"

    # Render result page
    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
