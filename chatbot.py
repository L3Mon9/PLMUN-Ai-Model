#python.exe .\chatbot.py
# =============================================================================
# ADVANCED HYBRID NLP CHATBOT — ALL-IN-ONE (THESIS FINAL VERSION)
# School: Pamantasan ng Lungsod ng Muntinlupa (PLMun)
# Features:
#   ✔ Rule-Based + Naive Bayes (Hybrid AI)
#   ✔ Human-like responses (entry + end messages)
#   ✔ Small talk (hi, hello, thanks, bye)
#   ✔ Context memory (follow-up support)
#   ✔ Confidence scoring (AI explanation)
#   ✔ Typo correction
#   ✔ Taglish support
#   ✔ Logging system (analytics)
# =============================================================================

import re
import csv
import sys
import os
import random
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# =============================================================================
# CONFIG
# =============================================================================

BOT_NAME = "PLMun Assistant 🤖"
LOG_FILE = "chat_logs.txt"
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

# =============================================================================
# MEMORY SYSTEM
# =============================================================================

conversation_memory = {
    "last_intent": None,
    "history": []
}

# =============================================================================
# LOAD DATASET
# =============================================================================

def load_dataset(filepath):
    texts, labels = [], []
    with open(filepath, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            texts.append(row["text"].strip())
            labels.append(row["intent"].strip())
    return texts, labels

# =============================================================================
# SPELL CORRECTION
# =============================================================================

def simple_spell_fix(text):
    corrections = {
        "hii": "hi",
        "helo": "hello",
        "enrol": "enroll",
        "schdule": "schedule",
        "documnts": "documents"
    }
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

# =============================================================================
# PREPROCESS
# =============================================================================

def preprocess(text):
    text = text.lower()
    text = simple_spell_fix(text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =============================================================================
# KEYWORD MAP
# =============================================================================

KEYWORD_MAP = {
    "greeting": ["hi", "hello", "hey", "kumusta", "musta"],
    "farewell": ["bye", "goodbye", "see you"],
    "thanks": ["thanks", "thank you", "salamat"],

    "enrollment": [
        "enroll", "mag enroll", "apply", "admission", "tuition"
    ],
    "schedule": [
        "schedule", "class", "oras", "kelan", "exam"
    ],
    "policies": [
        "policy", "rules", "attendance", "grading", "fail"
    ],
    "documents": [
        "tor", "certificate", "good moral", "registrar"
    ],
    "services": [
        "service", "library", "clinic", "guidance", "scholarship"
    ]
}

# =============================================================================
# RESPONSES (PLMUN BASED)
# =============================================================================

RESPONSES = {

    "greeting": [
        "Hi! 👋 Welcome to PLMun Assistant. How can I help you today?",
        "Hello! 😊 Ask me anything about enrollment, schedule, or school services."
    ],

    "farewell": [
        "Goodbye! 👋 Good luck sa studies mo!",
        "See you! Stay safe and study well!"
    ],

    "thanks": [
        "You're welcome! 😊",
        "No problem! Happy to help!"
    ],

    "enrollment": [
        "For PLMun enrollment, go to the Registrar or use the student portal. Bring your requirements like Form 138 and birth certificate.",
        "Enrollment usually happens before semester starts. Check official announcements for schedule."
    ],

    "schedule": [
        "Check your schedule via the student portal or Registrar.",
        "Academic calendar includes exams, holidays, and semester dates."
    ],

    "policies": [
        "PLMun follows rules on attendance, grading, and student conduct.",
        "Avoid cheating and misconduct — strict policies apply."
    ],

    "documents": [
        "Request TOR, certificates, and documents at the Registrar.",
        "Processing usually takes 3–5 working days."
    ],

    "services": [
        "PLMun offers services like library, clinic, guidance, and scholarships.",
        "Visit student services office for assistance."
    ],

    "unknown": [
        "Hmm 🤔 I didn’t understand that. Try asking about enrollment, schedule, or services.",
        "I'm not sure about that, but I can help with school-related questions 😊"
    ]
}

# =============================================================================
# RULE-BASED
# =============================================================================

def rule_based_classify(text):
    for intent, words in KEYWORD_MAP.items():
        for w in words:
            if w in text:
                return intent
    return None

# =============================================================================
# TRAIN MODEL
# =============================================================================

def train_model(texts, labels):
    processed = [preprocess(t) for t in texts]
    vectorizer = CountVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(processed)
    model = MultinomialNB()
    model.fit(X, labels)
    return vectorizer, model

# =============================================================================
# CONFIDENCE PREDICTION
# =============================================================================

def predict_with_confidence(text, vectorizer, model):
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    max_prob = max(probs)
    intent = model.classes_[probs.argmax()]
    return intent, max_prob

# =============================================================================
# RESPONSE
# =============================================================================

def get_response(intent):
    return random.choice(RESPONSES.get(intent, RESPONSES["unknown"]))

# =============================================================================
# LOGGING
# =============================================================================

def log_interaction(user, intent, method):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}]\nUser: {user}\nIntent: {intent}\nMethod: {method}\n\n")

# =============================================================================
# CHATBOT
# =============================================================================

def chatbot_respond(user_input, vectorizer, model):
    cleaned = preprocess(user_input)

    # FOLLOW-UP
    if cleaned in ["what about that", "tell me more"]:
        if conversation_memory["last_intent"]:
            intent = conversation_memory["last_intent"]
            return get_response(intent), intent, "Follow-Up"

    # RULE
    intent = rule_based_classify(cleaned)
    method = "Rule-Based"

    # ML
    if intent is None:
        intent, confidence = predict_with_confidence(cleaned, vectorizer, model)
        method = f"NaiveBayes ({confidence:.2f})"

        if confidence < 0.4:
            return "Can you rephrase that? 🤔", "unknown", method

    conversation_memory["last_intent"] = intent
    conversation_memory["history"].append(user_input)

    entry = random.choice([
        "Alright 👍",
        "Got it 👇",
        "Here's what I found:"
    ])

    end = random.choice([
        "Anything else?",
        "Ask me more anytime 😊",
        "I'm here to help 👍"
    ])

    response = f"{entry}\n{get_response(intent)}\n\n{end}"

    return response, intent, method

# =============================================================================
# UI
# =============================================================================

def banner():
    print("="*60)
    print("     PLMun AI CHATBOT 🤖 (THESIS SYSTEM)")
    print("="*60)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading AI model...")
    texts, labels = load_dataset(DATASET_FILE)
    vectorizer, model = train_model(texts, labels)

    banner()

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Bot: Goodbye 👋")
            break

        response, intent, method = chatbot_respond(user_input, vectorizer, model)

        log_interaction(user_input, intent, method)

        print(f"\nBot [{method}]:\n{response}\n")

# =============================================================================

if __name__ == "__main__":
    main()