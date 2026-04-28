from flask import Flask, render_template, request, jsonify
from chatbot import load_dataset, train_model, chatbot_respond

app = Flask(__name__)

# Load AI model once
print("Loading AI model...")
texts, labels = load_dataset("dataset.csv")
vectorizer, model = train_model(texts, labels)
print("✅ AI Model Loaded Successfully!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({
            "response": "Please type something.",
            "intent": "unknown",
            "method": "none"
        })

    response, intent, method = chatbot_respond(user_input, vectorizer, model)

    return jsonify({
        "response": response,
        "intent": intent,
        "method": method
    })

# IMPORTANT for Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)