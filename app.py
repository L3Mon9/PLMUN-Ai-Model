# =============================================================================
# FLASK WEB APP — AI Institutional Support Chatbot
# PLMun — Pamantasan ng Lungsod ng Muntinlupa
# Thesis: Development of an AI-Powered Chatbot Using NLP and Intent
#         Classification for Institutional Inquiry and Student Support Services
# =============================================================================
# ALGORITHMS USED:
#   1. Rule-Based Keyword Matching     — Layer 1 (Primary Classifier)
#   2. Multinomial Naive Bayes         — Layer 2 (ML Fallback Classifier)
#   3. TF-IDF Vectorization            — Feature Extraction (smarter than BoW)
#   4. N-Gram Language Model (1-2)     — Captures word pairs for better context
# =============================================================================

import re
import csv
import os
import random
import time
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "dataset.csv")

# =============================================================================
# 1. LOAD DATASET
# =============================================================================

def load_dataset(filepath):
    texts, labels = [], []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"].strip())
            labels.append(row["intent"].strip())
    return texts, labels

# =============================================================================
# 2. NLP PREPROCESSING
# =============================================================================

def preprocess(text):
    """
    NLP Preprocessing Pipeline:
    Step 1 - Lowercasing
    Step 2 - Remove punctuation and special characters
    Step 3 - Normalize whitespace
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =============================================================================
# 3. RULE-BASED KEYWORD MAP (Layer 1 — Primary Classifier)
# =============================================================================

KEYWORD_MAP = {
    "greeting": [
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "howdy", "greetings", "good day",
        "hi there", "hello there", "hey there", "kumusta", "musta",
        "anyone there", "is anyone available"
    ],
    "farewell": [
        "bye", "goodbye", "see you", "take care", "farewell",
        "see ya", "im done", "thats all", "nothing else", "ok bye",
        "leaving now", "have to go", "cya", "done na", "ok lang na"
    ],
    "thanks": [
        "thank you", "thanks", "thank u", "ty", "salamat",
        "maraming salamat", "i appreciate", "that was helpful",
        "got it thanks", "noted thanks", "ok thanks"
    ],
    "enrollment": [
        "enroll", "enrollment", "enrolment", "admission", "tuition",
        "register", "registration", "transferee", "applicant",
        "apply", "application", "incoming", "mag enroll", "mag apply",
        "enrollment period", "enrollment deadline", "enrollment requirements",
        "enroll late", "online enrollment", "enrollment fee", "add subjects"
    ],
    "schedule": [
        "schedule", "class", "classes", "semester", "academic calendar",
        "timetable", "calendar", "holiday", "semestral break",
        "midterm", "final exam", "first day of class", "make up class",
        "class suspended", "last day of class", "school day",
        "kelan ang", "kailan magsisimula", "may pasok", "anong oras"
    ],
    "policies": [
        "policy", "policies", "rules", "grade", "grading", "attendance",
        "absent", "absences", "fail", "failing", "shift", "dropout",
        "dress code", "retention", "conduct", "appeal", "incomplete",
        "cheating", "misconduct", "leave of absence", "residency",
        "passing grade", "drop subject", "bagsak", "bumagsak",
        "pano kung bumagsak", "ilan ang allowed", "probation",
        "grading scale", "final grade", "academic probation"
    ],
    "documents": [
        "tor", "transcript", "certificate", "registrar", "document",
        "diploma", "clearance", "form 137", "credentials", "release",
        "authorization", "good moral", "claim", "honorable dismissal",
        "proof of enrollment", "copy of grades", "paano kumuha",
        "saan kukuha", "bayad sa dokumento", "official record"
    ],
    "services": [
        "service", "services", "guidance", "scholarship", "financial aid",
        "support", "portal", "clinic", "library", "dormitory",
        "cashier", "canteen", "organization", "lost and found",
        "counseling", "school nurse", "student org", "financial help",
        "lost id", "pay tuition", "mag apply ng scholarship",
        "nasaan ang guidance", "saan ang cashier", "may scholarship",
        "helpdesk", "hotline", "complaint", "student affairs"
    ]
}

def rule_based_classify(text):
    """Layer 1: Rule-Based Keyword Matching — O(n*m) pattern matching."""
    for intent, keywords in KEYWORD_MAP.items():
        for keyword in keywords:
            if keyword in text:
                return intent
    return None

# =============================================================================
# 4. RESPONSES (Multiple per intent for natural variation)
# =============================================================================

RESPONSES = {
    "greeting": [
        "Hello! Welcome to the PLMun Student Support Chatbot. I'm here to help you with enrollment, schedules, policies, documents, and student services. What can I assist you with today?",
        "Hi there! Great to hear from you. Feel free to ask me anything about enrollment, your schedule, school policies, document requests, or available student services at PLMun!",
        "Good day, Iskolar! How can I help you today? Whether it's about enrollment, schedules, policies, documents, or services — just type your question.",
        "Hello! I'm your PLMun institutional support assistant. You can ask me about enrollment requirements, class schedules, school rules, document requests, or student services. How may I help?",
    ],
    "farewell": [
        "Thank you for reaching out! Have a great day and good luck with your studies at PLMun! Feel free to come back anytime.",
        "Goodbye, Iskolar! It was my pleasure to assist you. Take care and stay safe!",
        "See you! Don't hesitate to come back if you have more questions. Goodluck sa studies!",
        "Take care! Wishing you all the best in your academic journey at PLMun. Goodbye!",
    ],
    "thanks": [
        "You're welcome, Iskolar! If you have any more questions, feel free to ask anytime!",
        "Glad I could help! Is there anything else you'd like to know?",
        "No problem at all! That's what I'm here for. Anything else I can assist you with?",
        "Happy to help! Don't hesitate to ask if you need more information.",
    ],
    "enrollment": [
        "For enrollment at PLMun, please visit the Registrar's Office during the official enrollment period. Bring your report card, birth certificate, and medical certificate. Online enrollment may also be available through the Student Portal. Contact the Admissions Office for more details.",
        "Enrollment is processed at the PLMun Registrar's Office. Prepare your required documents: report card, birth certificate, and medical certificate. Check the school website or Student Portal for the schedule and procedures. Transferees may need additional credentials.",
        "To enroll at PLMun, visit the Registrar's Office within the enrollment period and submit required documents. If you're a new student, bring your Form 138/137, birth certificate, and medical certificate. For online enrollment, access the Student Portal at the PLMun website.",
    ],
    "schedule": [
        "Your class schedule is available through the PLMun Student Portal or at the Registrar's Office. The academic calendar — including semester start/end dates, exam periods, and holidays — is posted on the school bulletin board and website. Coordinate with your adviser for adjustments.",
        "Check your class schedule anytime through the PLMun Student Portal! The academic calendar with midterms, finals, and semestral breaks is on the official website. For schedule changes, talk to your academic adviser or department head.",
        "Class schedules are posted at the Registrar's Office and through the Student Portal. Important dates like the first day of class, midterm week, final exams, and semestral breaks are in the academic calendar on the PLMun website.",
    ],
    "policies": [
        "PLMun's academic policies are in the Student Handbook given during enrollment. This covers the grading system, attendance policy, retention requirements, and code of conduct. For concerns, visit the Office of Academic Affairs or your academic adviser.",
        "All academic rules are in the PLMun Student Handbook. Key policies: grading system, maximum absences allowed, dress code, retention policy, and academic conduct. For grade appeals or attendance issues, visit the Office of Academic Affairs.",
        "Refer to your PLMun Student Handbook for all school policies — from passing grades and attendance rules to shifting procedures and the code of conduct. Your academic adviser or the Office of Academic Affairs can help clarify specific policies.",
    ],
    "documents": [
        "Request official documents at the PLMun Registrar's Office. Available documents include TOR, Certificate of Enrollment, Good Moral Character, Form 137, and Honorable Dismissal. Processing takes 3–5 working days. Bring a valid ID and pay the document fee at the Cashier's Office.",
        "For document requests at PLMun, go to the Registrar's Office with a valid ID. You can request your TOR, Certificate of Enrollment, Good Moral, school clearance, and more. Processing time is 3–5 working days. Pay the corresponding fee at the Cashier. Online requests may be available on the Student Portal.",
        "The PLMun Registrar's Office handles all official document requests. Bring a valid ID and settle the document fee at the Cashier's Office. Processing takes approximately 3–5 working days. For authorization of release, submit a letter with a photocopy of your ID.",
    ],
    "services": [
        "PLMun offers: Guidance and Counseling Office, Library, Medical and Dental Clinic, Cashier's Office, and Student Affairs Office. Scholarship and financial assistance programs are available — visit the Student Affairs Office for eligibility requirements and application.",
        "Student services at PLMun include the Guidance Office, Library, Medical Clinic, Cashier, and the Student Affairs Office. For scholarships and financial aid, inquire at the Student Affairs Office. Access some services through the PLMun Student Portal.",
        "PLMun student services: Library for academic resources, Guidance Office for personal concerns, Medical Clinic for health needs, and Student Affairs for scholarships. For lost IDs, report to the Security Office or Student Affairs. Contact the school hotline for other concerns.",
    ],
    "unknown": [
        "I'm sorry, I didn't quite catch that. I'm the PLMun Student Support Chatbot and I work best with questions about enrollment, schedules, policies, documents, or student services. Could you rephrase your question?",
        "Hmm, that's a bit outside what I can help with. I'm designed for PLMun institutional inquiries — enrollment, class schedules, school policies, document requests, and student services. Try asking about those!",
        "I appreciate you reaching out! I can only assist with PLMun school-related concerns such as enrollment, schedules, policies, documents, and services. Feel free to ask about any of those topics!",
        "That one's a bit tricky for me! I'm the PLMun Student Support Chatbot — best at answering questions about enrollment, class schedules, school policies, document requests, and student services.",
    ]
}

def get_response(intent):
    return random.choice(RESPONSES.get(intent, RESPONSES["unknown"]))

# =============================================================================
# 5. TRAIN MODEL — TF-IDF + Multinomial Naive Bayes Pipeline (Layer 2)
# =============================================================================

def train_model(texts, labels):
    """
    ML Pipeline:
    Step 1 — TF-IDF Vectorization (smarter than raw Bag-of-Words)
             Assigns higher weight to rare but meaningful words
    Step 2 — Multinomial Naive Bayes Classification
             Uses Bayes' theorem: P(intent|words) ∝ P(words|intent) * P(intent)
    N-Gram range (1,2): captures unigrams AND bigrams for context
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            analyzer='word'
        )),
        ('nb', MultinomialNB(alpha=0.3))
    ])
    processed = [preprocess(t) for t in texts]
    pipeline.fit(processed, labels)
    return pipeline

def evaluate_model(texts, labels, pipeline):
    """Full evaluation: accuracy, precision, recall, F1, confusion matrix, cross-val."""
    processed = [preprocess(t) for t in texts]
    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels
    )
    eval_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=1)),
        ('nb', MultinomialNB(alpha=0.3))
    ])
    eval_pipeline.fit(X_train, y_train)
    y_pred = eval_pipeline.predict(X_test)

    intents = sorted(set(labels))
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=intents)

    cv_scores = cross_val_score(eval_pipeline, processed, labels, cv=5, scoring='accuracy')

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "cross_val_mean": round(cv_scores.mean() * 100, 2),
        "cross_val_std": round(cv_scores.std() * 100, 2),
        "cv_scores": [round(s * 100, 2) for s in cv_scores],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_samples": len(labels),
        "intents": intents,
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_test, y_pred, target_names=intents, output_dict=True)
    }

# =============================================================================
# 6. STARTUP — Load data and train
# =============================================================================

print("Loading dataset...", end=" ")
texts, labels = load_dataset(DATASET_FILE)
print(f"Done. ({len(texts)} samples)")

print("Training model...", end=" ")
pipeline = train_model(texts, labels)
print("Done.")

eval_results = evaluate_model(texts, labels, pipeline)
print(f"Model Accuracy : {eval_results['accuracy']}%")
print(f"F1 Score       : {eval_results['f1_score']}%")
print(f"Cross-Val Mean : {eval_results['cross_val_mean']}% ± {eval_results['cross_val_std']}%")
print("✅ PLMun Chatbot Ready!")

# =============================================================================
# 7. ROUTES
# =============================================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please type a message.", "intent": "unknown", "method": "none", "confidence": 0})

    start = time.time()
    cleaned = preprocess(user_input)
    intent = rule_based_classify(cleaned)
    method = "Rule-Based"
    confidence = 100.0

    if intent is None:
        probs = pipeline.predict_proba([cleaned])[0]
        intent_idx = probs.argmax()
        intent = pipeline.classes_[intent_idx]
        confidence = round(float(probs[intent_idx]) * 100, 1)
        method = "Naive Bayes"

    response_time = round((time.time() - start) * 1000, 2)
    response = get_response(intent)

    return jsonify({
        "response": response,
        "intent": intent,
        "method": method,
        "confidence": confidence,
        "response_time_ms": response_time
    })

@app.route("/eval", methods=["GET"])
def get_eval():
    return jsonify(eval_results)

@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify({
        "total_samples": len(texts),
        "total_intents": len(set(labels)),
        "intents": sorted(set(labels)),
        "algorithm": "Hybrid: Rule-Based + TF-IDF + Multinomial Naive Bayes",
        "accuracy": eval_results["accuracy"],
        "f1_score": eval_results["f1_score"]
    })

# =============================================================================
# 8. RUN
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True)