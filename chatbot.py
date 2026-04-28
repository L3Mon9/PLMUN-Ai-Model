# =============================================================================
# FLASK WEB APP — AI Institutional Support Chatbot (ENHANCED v2.0)
# PLMun — Pamantasan ng Lungsod ng Muntinlupa
# =============================================================================

import re, csv, os, time, random
from flask import Flask, render_template, request, jsonify, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
import numpy as np
from collections import Counter
from rapidfuzz import fuzz

app = Flask(__name__)
app.secret_key = "plmun_chatbot_secret_2024"   # needed for session (context memory)

DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

# =============================================================================
# DATASET LOADER & PREPROCESSOR
# =============================================================================

def load_dataset(filepath):
    texts, labels = [], []
    with open(filepath, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            texts.append(row["text"].strip())
            labels.append(row["intent"].strip())
    return texts, labels

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# =============================================================================
# ENHANCED SLANG / TAGLISH MAP
# =============================================================================

SLANG_MAP = {
    # General Taglish / shorthand
    "pls": "please",
    "plz": "please",
    "pls help": "please help",
    "saan": "where",
    "san": "where",
    "ano": "what",
    "paano": "how",
    "kailan": "when",
    "pwede": "can i",
    "pwde": "can i",
    "bakit": "why",
    "sino": "who",
    "kelan": "when",
    "po": "",
    "naman": "",
    "nga": "",
    "ba": "",
    "kaya": "",
    "lang": "",
    "narin": "also",
    "din": "also",
    "daw": "",
    "raw": "",
    "yung": "the",
    "ung": "the",
    "mga": "some",
    "nung": "when",
    "pag": "if",
    "kapag": "if",
    "kasi": "because",
    "talaga": "really",
    "naman": "",
    "muna": "first",
    "na": "",
    "pa": "still",
    "ay": "is",
    "yun": "that",
    "yan": "that",
    "ito": "this",
    "dito": "here",
    "doon": "there",

    # Common shorthand
    "sched": "schedule",
    "req": "requirements",
    "reqs": "requirements",
    "doc": "document",
    "docs": "documents",
    "docu": "document",
    "sec": "section",
    "sem": "semester",
    "enrol": "enroll",
    "enrolment": "enrollment",
    "dept": "department",
    "prof": "professor",
    "profs": "professors",
    "subj": "subject",
    "subjs": "subjects",
    "subje": "subject",
    "grade": "grade",
    "tor": "transcript of records",
    "coe": "certificate of enrollment",
    "loa": "leave of absence",
    "hd": "honorable dismissal",
    "cog": "certificate of grades",
    "pcat": "entrance exam",
    "inc": "incomplete grade",

    # Taglish phrases
    "paano mag enroll": "how to enroll",
    "paano kumuha": "how to get",
    "paano mag drop": "how to drop",
    "paano mag shift": "how to shift",
    "paano mag loa": "how to take leave of absence",
    "paano mag apply scholarship": "how to apply for scholarship",
    "paano makita": "how to view",
    "paano mag request": "how to request",
    "section ko": "my section",
    "block ko": "my section",
    "sched ko": "my schedule",
    "grade ko": "my grade",
    "email pls": "email",
    "ano ba": "what is",
    "ano ang": "what is",
    "kailan ang": "when is",
    "saan ang": "where is",
    "pwede bang": "can i",
    "may pasok ba": "is there class",
    "walang pasok": "class suspended",
    "mag aaral": "student",
    "mag-aaral": "student",
    "estudyante": "student",
    "bagong estudyante": "new student",
    "pasukan": "first day of class",
    "pampaaralan": "school",
    "titser": "teacher",
    "guro": "teacher",
    "klase": "class",
    "pumunta": "go to",
    "lugar": "place",
    "bayad": "payment",
    "libre": "free",
    "libre ba": "is it free",
    "magkano": "how much",
    "magbayad": "pay",
    "sumali": "join",
    "mag-apply": "apply",
    "mag apply": "apply",
    "kumuha": "get",
    "humingi": "request",
    "tanungin": "ask",
    "malaman": "know",
    "makita": "see",
    "makuha": "get",
    "irequest": "request",
    "i-request": "request",
    "icheck": "check",
    "i-check": "check",
    "isubmit": "submit",
    "i-submit": "submit",
    "nawala": "lost",
    "napalipas": "missed",
    "nalipas": "missed",
    "absentehin": "absent",
    "ma-absent": "be absent",
    "mag-absent": "be absent",
    "umabsent": "be absent",
    "bagsak": "failed",
    "bumagsak": "failed",
    "sumobra": "exceeded",
    "ilang": "how many",
    "ilan": "how many",
    "iilan": "how many",
}

# =============================================================================
# INTENT REFINEMENT PATTERNS
# Maps modifier phrases to sub-intent suffix
# =============================================================================

INTENT_REFINERS = [
    # Permission / allowed
    (["pwede bang", "allowed ba", "okay ba", "puwede ba", "puwede bang", "can i", "may i", "is it okay", "is it allowed", "permitted"], "_allowed"),
    # Importance / why
    (["mahalaga", "bakit mahalaga", "why important", "importance of", "why do i need", "kailangan ba", "need ba", "do i need"], "_importance"),
    # How to / steps
    (["paano", "how to", "how do i", "how can i", "steps", "procedure", "proseso", "process", "guide", "tutorial"], "_steps"),
    # What is / definition
    (["ano ang", "what is", "ano ba ang", "define", "meaning of", "ibig sabihin", "what does", "anong ibig sabihin"], "_definition"),
    # When / deadline
    (["kailan", "when is", "when can i", "deadline", "due date", "petsa", "last day", "pano malaman kung kailan"], "_when"),
    # Where / location
    (["saan", "where is", "where can i", "location", "lugar", "office", "pupuntahan"], "_where"),
    # How much / cost
    (["magkano", "how much", "bayad", "cost", "fee", "libre ba", "is it free", "price"], "_cost"),
    # Who / contact
    (["sino", "who", "who do i", "who should", "sino ang"], "_who"),
]

# =============================================================================
# SPECIFIC KEYWORD RULES — maps exact topics to specific intents
# =============================================================================

SPECIFIC_RULES = [
    # 🔥 PRIORITY — ENROLLMENT DATE (UNA DAPAT)
    (["kelan enrollment", "kailan enrollment", "when enrollment",
  "enrollment date", "enrollment schedule",
  "second sem enrollment", "2nd sem enrollment",
  "second semester enrollment", "enrollment second sem",
  "kelan second sem", "kailan second sem",
  "enrollment second semester"], "schedule_enrollment"),

    # ======================
    # 📅 SCHEDULE
    # ======================
    (["check schedule", "view schedule", "see schedule", "how to check my schedule", "student portal schedule"], "schedule_check"),
    (["semester start", "start of semester", "when does semester start", "pasukan", "first day"], "schedule_semester_start"),
    (["midterm", "midterms", "midterm exam", "when is midterm"], "schedule_midterm"),
    (["final exam", "finals", "final exams", "when is final"], "schedule_finals"),
    (["semestral break", "sem break", "vacation", "christmas vacation", "school break"], "schedule_break"),
    (["class suspended", "class suspension", "walang pasok", "may pasok"], "schedule_suspension"),
    (["dropping period", "add drop", "adding dropping", "drop deadline"], "schedule_dropping"),
    (["academic calendar", "school calendar", "school year"], "schedule_calendar"),
    (["schedule", "class schedule", "timetable", "anong oras", "what time", "oras ng klase", "klase", "class time"], "schedule_general"),

    # ======================
    # 🎓 ENROLLMENT
    # ======================
    (["transferee", "transfer student", "mag transfer"], "enrollment_transferee"),
    (["freshmen", "freshman", "new student", "first year", "bagong estudyante"], "enrollment_freshmen"),
    (["pcat", "college admission test", "entrance exam", "entrance test"], "enrollment_pcat"),
    (["courses offered", "course available", "what course", "anong course", "programs offered", "what program", "available course"], "enrollment_courses"),
    (["tuition fee", "magkano", "how much", "payment", "installment", "miscellaneous fee"], "enrollment_tuition"),
    (["enroll online", "online enrollment", "online enroll"], "enrollment_online"),
    (["enroll late", "late enrollment", "nalipas enrollment"], "enrollment_late"),
    (["requirements for enrollment", "enrollment requirements", "documents needed for enrollment"], "enrollment_requirements"),

    # 🔻 LAST (GENERAL)
    (["enroll", "enrollment", "enrolment", "mag enroll", "how to enroll", "paano mag enroll"], "enrollment_general"),

    # POLICIES
    (["grading system", "grading scale", "how grades", "paano ang grades", "grade equivalent"], "policies_grading"),
    (["passing grade", "passing score", "bakit ako bagsak", "failed", "bumagsak", "what is passing"], "policies_passing"),
    (["absences", "absent", "umabsent", "how many absences", "ilan absent", "attendance rules", "attendance policy", "allowed absences", "ma-absent"], "policies_attendance"),
    (["inc grade", "incomplete grade", "pano alisin ang inc"], "policies_inc"),
    (["drop subject", "dropping subject", "mag drop", "pwede bang mag drop"], "policies_drop"),
    (["appeal grade", "grade appeal", "contest grade", "disagree with grade", "mali grade ko"], "policies_appeal"),
    (["shift", "shifting course", "mag shift", "change course", "change program"], "policies_shift"),
    (["leave of absence", "loa", "mag loa", "mag leave"], "policies_loa"),
    (["latin honor", "cum laude", "magna cum laude", "summa cum laude", "dean's list", "deans list", "honors"], "policies_honors"),
    (["cheating", "plagiarism", "academic dishonesty", "kopya"], "policies_cheating"),
    (["dress code", "uniform", "what to wear", "school uniform"], "policies_dressCode"),
    (["retain", "retention", "academic probation", "probation", "dismissed", "dismissal"], "policies_retention"),
    (["maximum residency", "max residency", "how many years", "maximum years"], "policies_residency"),
    (["policy", "policies", "rules", "academic rules", "school rules"], "policies_general"),
    (["section", "ano section ko", "what is my section", "block ko", "class section", "section ko"], "section_info"),

    # DOCUMENTS
    (["tor", "transcript of records", "transcript", "kumuha ng tor", "request tor"], "documents_tor"),
    (["certificate of enrollment", "proof of enrollment", "enrollment certificate", "certificate of registration"], "documents_coe"),
    (["good moral", "good moral character", "certificate of good moral"], "documents_goodmoral"),
    (["diploma", "lost diploma", "diploma replacement", "nawala diploma", "wala diploma"], "documents_diploma"),
    (["clearance", "school clearance", "get clearance"], "documents_clearance"),
    (["form 137", "f137"], "documents_form137"),
    (["honorable dismissal", "transfer credential"], "documents_hd"),
    (["board exam", "licensure exam", "board application", "prc"], "documents_boardexam"),
    (["authorization", "proxy claim", "someone claim", "ibang tao kumuha"], "documents_proxy"),
    (["document fee", "bayad sa dokumento", "how much document", "magkano document"], "documents_fee"),
    (["doc", "docs", "docu", "documents", "document", "certificate", "request document", "official record", "registrar", "paano kumuha", "papeles"], "documents_general"),

    # SERVICES
    (["plmun ng bayan", "city scholarship", "muntinlupa scholarship", "plmun scholarship"], "services_PLmunnbayan"),
    (["scholarship", "financial aid", "financial assistance", "may scholarship", "apply scholarship", "mag apply scholarship"], "services_scholarship"),
    (["student id", "id replacement", "lost id", "replace id", "new id", "get id",
  "kuha id", "kumuha ng id", "paano kumuha ng id",
  "how to get id", "how to get student id",
  "id paano", "id process", "student id paano"], "services_id"),
    (["student portal", "portal", "login portal", "cannot login", "reset password", "forgot password"], "services_portal"),
    (["guidance", "counseling", "counselor", "guidance office"], "services_guidance"),
    (["library", "elibrary", "borrow book", "library card", "library hours"], "services_library"),
    (["clinic", "medical", "school nurse", "dental", "health services"], "services_clinic"),
    (["cashier", "treasury", "saan magbayad", "where to pay", "payment office"], "services_cashier"),
    (["contact", "telephone", "email plmun", "hotline", "how to contact", "plmun number"], "services_contact"),
    (["location", "address", "where is plmun", "campus", "nasaan ang plmun"], "services_location"),
    (["student organization", "student org", "join org", "join club", "student council"], "services_org"),
    (["complaint", "report concern", "report problem", "reklamo"], "services_complaint"),
    (["service", "services", "student services", "what services"], "services_general"),

    # CONVERSATIONAL
    (["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "kumusta", "musta", "good day", "howdy"], "greeting"),
    (["bye", "goodbye", "see you", "take care", "im done", "thats all", "ok bye", "cya", "sige na", "sige"], "farewell"),
    (["thank you", "thanks", "thank u", "salamat", "maraming salamat", "i appreciate", "got it thanks", "ok thanks", "ty"], "thanks"),
]

# =============================================================================
# TOPIC DETECTION MAP — for context tracking & refinement
# =============================================================================

TOPIC_MAP = {
    "enrollment": ["enrollment_general", "enrollment_freshmen", "enrollment_transferee",
                   "enrollment_pcat", "enrollment_tuition", "enrollment_online",
                   "enrollment_late", "enrollment_requirements", "enrollment_courses"],
    "schedule": ["schedule_general", "schedule_check", "schedule_finals", "schedule_midterm",
                 "schedule_break", "schedule_suspension", "schedule_dropping",
                 "schedule_calendar", "schedule_semester_start"],
    "policies": ["policies_general", "policies_grading", "policies_passing",
                 "policies_attendance", "policies_inc", "policies_drop", "policies_appeal",
                 "policies_shift", "policies_loa", "policies_honors", "policies_cheating",
                 "policies_dressCode", "policies_retention", "policies_residency"],
    "documents": ["documents_general", "documents_tor", "documents_coe", "documents_goodmoral",
                  "documents_diploma", "documents_clearance", "documents_form137",
                  "documents_hd", "documents_boardexam", "documents_proxy", "documents_fee"],
    "services": ["services_general", "services_scholarship", "services_PLmunnbayan",
                 "services_id", "services_portal", "services_guidance", "services_library",
                 "services_clinic", "services_cashier", "services_contact",
                 "services_location", "services_org", "services_complaint"],
}

def get_topic(intent):
    for topic, intents in TOPIC_MAP.items():
        if intent in intents:
            return topic
    return None

# =============================================================================
# CONTEXT MEMORY — session-based last intent/topic tracking
# =============================================================================

def get_context():
    return {
        "last_intent": session.get("last_intent"),
        "last_topic": session.get("last_topic"),
        "turn_count": session.get("turn_count", 0),
    }

def update_context(intent):
    session["last_intent"] = intent
    session["last_topic"] = get_topic(intent)
    session["turn_count"] = session.get("turn_count", 0) + 1

# =============================================================================
# FOLLOW-UP DETECTION — catches short follow-up messages
# =============================================================================

FOLLOWUP_SIGNALS = [
    "ilan", "how many", "magkano", "how much", "saan", "where", "kailan", "when",
    "ano pa", "what else", "then what", "and then", "next", "after that",
    "paano", "how", "bakit", "why", "sino", "who", "pwede ba", "can i",
    "ano", "what", "ok", "really", "ganun", "talaga", "sure", "totoo",
    "example", "halimbawa", "details", "more info", "tell me more",
]

def is_followup(user_input, context):
    """Detect if this is a short follow-up referencing a previous topic."""
    if not context.get("last_intent"):
        return False
    cleaned = user_input.lower().strip()
    # Short message is likely a follow-up
    if len(cleaned.split()) <= 4:
        for signal in FOLLOWUP_SIGNALS:
            if signal in cleaned:
                return True
    return False

# =============================================================================
# NORMALIZATION — Taglish + slang → clean English
# =============================================================================

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Apply slang map (longest phrases first to avoid partial replacements)
    for k in sorted(SLANG_MAP.keys(), key=len, reverse=True):
        text = re.sub(r'\b' + re.escape(k) + r'\b', SLANG_MAP[k], text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =============================================================================
# INTENT REFINEMENT — adds sub-intent suffix based on question type
# =============================================================================

def refine_intent(base_intent, user_input):
    """
    Detect if the user's phrasing indicates a sub-intent (allowed, steps, etc.)
    and return a refined intent string. Falls back to base_intent if no match.
    """
    normalized = normalize_text(user_input)
    for patterns, suffix in INTENT_REFINERS:
        for pattern in patterns:
            if pattern in normalized:
                refined = base_intent + suffix
                # Only use refined if we have a response for it
                if refined in RESPONSES:
                    return refined
                break
    return base_intent

# =============================================================================
# SMART CLASSIFY — Rule-based + Fuzzy + ML fallback
# =============================================================================

def smart_classify(user_input, context=None):
    normalized = normalize_text(user_input)

    # 🔥 0. HARD OVERRIDE (critical intents)
    if "id" in normalized:
        return "services_id", "Rule-Override", 100.0

    # 🔹 1. EXACT MATCH (improved)
    for keywords, intent in SPECIFIC_RULES:
        for kw in keywords:
            if all(word in normalized for word in kw.split()):
                return intent, "Rule-Based", 100.0

    # 🔹 2. FUZZY MATCH
    best_intent = None
    best_score = 0
    for keywords, intent in SPECIFIC_RULES:
        for kw in keywords:
            score = fuzz.partial_ratio(normalized, kw)
            if score > best_score:
                best_score = score
                best_intent = intent

    if best_score >= 80:
        return best_intent, "Fuzzy", round(best_score, 1)

    # 🔹 3. CONTEXT FOLLOW-UP
    if context and is_followup(user_input, context):
        last_intent = context.get("last_intent")
        if last_intent:
            refined = refine_intent(last_intent, user_input)
            if refined != last_intent:
                return refined, "Context-Refined", 90.0
            return last_intent, "Context-Followup", 85.0

    # 🔹 4. ML MODEL
    cleaned = preprocess(user_input)
    probs = pipeline.predict_proba([cleaned])[0]
    idx = probs.argmax()
    confidence = round(float(probs[idx]) * 100, 1)

    if confidence >= 60:
        return pipeline.classes_[idx], "Naive Bayes", confidence

    # 🔹 5. FALLBACK
    return "unknown", "Fallback", confidence

# =============================================================================
# SMART REPHRASING — acknowledges what the user asked
# =============================================================================

REPHRASE_TEMPLATES = {
    "enrollment_general":       ["Ah, tinatanong mo kung paano mag-enroll sa PLMun 😊", "I see, you want to know about enrollment!"],
    "enrollment_freshmen":      ["So, freshman ka pala! Let me help you out 🎓", "Ah, bagong estudyante! Here's what you need to know:"],
    "enrollment_transferee":    ["Transferee ka pala! Okay, here are the details 😊", "So you're transferring to PLMun! Let me explain:"],
    "enrollment_tuition":       ["Ah, about sa tuition fee ang tanong mo 💰", "You're asking about the cost of studying here!"],
    "enrollment_pcat":          ["Ah, about sa entrance exam (PCAT) ang tanong mo 📝", "So you want to know about the PLMun entrance test!"],
    "schedule_general":         ["So, schedule-related ang tanong mo 📅", "Ah, gusto mo malaman ang class schedule mo!"],
    "schedule_semester_start":  ["Ah, tinatanong mo kung kailan magsisimula ang klase 📅", "You want to know the first day of class!"],
    "schedule_finals":          ["So, about final exams ang tanong mo 📚", "Ah, finals na pala! Let me check that for you:"],
    "policies_attendance":      ["Ah, about sa attendance policy ang tanong mo 📋", "You're asking about absences and attendance rules!"],
    "policies_grading":         ["So, gusto mo malaman ang grading system? 📊", "Ah, about sa grades ang tanong mo!"],
    "policies_drop":            ["Ah, about sa dropping subject ang tanong mo 📌", "You want to know how to drop a subject!"],
    "policies_shift":           ["So, nag-iisip kang mag-shift ng course? 🔄", "Ah, you're asking about how to change your program!"],
    "policies_loa":             ["Ah, about sa Leave of Absence ang tanong mo 📄", "You want to take a break from school — here's what to do:"],
    "documents_tor":            ["Ah, need mo ng TOR! Here's how to get it 📄", "So you're requesting a Transcript of Records!"],
    "documents_coe":            ["Ah, COE ang kailangan mo! Simple lang 😊", "You need a Certificate of Enrollment — no problem!"],
    "documents_goodmoral":      ["Ah, Good Moral Certificate ang kailangan mo 😊", "So you need a certificate of good moral character!"],
    "services_scholarship":     ["Ah, interested ka sa scholarship! 🎓", "You're asking about financial assistance — great question!"],
    "services_portal":          ["Ah, may issue sa Student Portal? Let me help 💻", "You're having trouble with the portal — let's fix that!"],
    "services_guidance":        ["Ah, about sa Guidance Office ang tanong mo 💬", "You want to reach the counseling office — they're always ready to help!"],
    "services_id":              ["Ah, about sa Student ID ang tanong mo 🪪", "You need help with your student ID!"],
    "section_info":             ["Ah, gusto mo malaman ang section mo 📘", "You're asking about your class section!"],
    "greeting":                 ["Hey! 👋", "Hi there! 😊"],
    "thanks":                   ["Glad to help! 😊", "Anytime! 💚"],
    "farewell":                 ["Sige, ingat! 👋", "Take care! 💚"],
}

def get_rephrase(intent, user_input):
    """Return a short acknowledgment line for the detected intent."""
    templates = REPHRASE_TEMPLATES.get(intent)
    if templates:
        return random.choice(templates)
    # Generic fallback rephrase
    generics = [
        f'So you\'re asking about "{user_input.strip()}" 😊',
        f'Got it! Let me help you with that.',
        f'Ah, I think I understand what you\'re asking about!',
        f'Sure! Here\'s what I know about that:',
    ]
    return random.choice(generics)

# =============================================================================
# RESPONSES — Multiple variants per intent for natural variation
# =============================================================================

RESPONSES = {

# ── CONVERSATIONAL ──────────────────────────────────────────────────────────
"schedule_enrollment": [
"""📅 Enrollment Schedule

Ah, ang tanong mo is kung kailan ang enrollment 😊

📌 Usually:
• 1st Semester Enrollment → July–August  
• 2nd Semester Enrollment → December–January  

📌 Note:
• Exact dates vary every school year  
• Check Student Portal or official announcements  

👉 Para sure:
📞 02-8248-9161  
🌐 www.plmun.edu.ph"""
],


"greeting": [
"""👋 Hi! Kumusta ka? 😊

Ako si PLMun Bot — nandito ako para sagutin ang mga katanungan mo tungkol sa:
• 🎓 Enrollment
• 📅 Schedules
• 📚 School Policies
• 📄 Documents
• 🏫 Student Services

Ano ang maitutulong ko sa iyo? 💚""",

"""Hello there! 😊 Welcome!

Sino ba ako? Ako ang PLMun Student Assistant Bot!
Ask me anything about:
• Enrollment process
• Class schedules
• Academic policies
• Document requests
• Student services

Sige, tanong ka lang! 💚""",

"""Hey! Good to see you here 👋

I'm your friendly PLMun chatbot!
Makakatulong ako sa mga katanungan tungkol sa enrollment, schedule, policies, documents, at services.

What can I help you with today? 😊"""
],

"farewell": [
"""👋 Sige, ingat ka!

Good luck sa studies mo sa PLMun! 💚
Kung may tanong ka ulit anytime — nandito lang ako. Ciao! 😊""",

"""Take care! 😊

Remember: magtayo ng sariling goals, mag-aral nang mabuti, at lagi kang magtanong kapag di ka sigurado! 💪

Babalik ka naman ha! 💚""",

"""Ingat lagi! 👋

Kung may kailangan kang malaman about PLMun, bumalik ka lang dito anytime. 
Good luck sa lahat ng pinagkakaabalahan mo! 💚😊"""
],

"thanks": [
"""Walang anuman! 😊

Lagi akong nandito para sagutin ang mga tanong mo. 
Mag-enjoy sa pag-aaral sa PLMun! 💚""",

"""Haha, salamat rin! 😄

Sana nakatulong ako nang konti. Kung may iba kang katanungan — huwag mahiyang magtanong! 💚""",

"""No problem at all! 😊

Yun ang trabaho ko — tumulong sa mga estudyante ng PLMun! 
Ask ka lang ulit anytime 💚"""
],

# ── ENROLLMENT ──────────────────────────────────────────────────────────────

"enrollment_general": [
"""🎓 Enrollment Guide

Para mag-enroll sa PLMun, ito ang proseso:

📌 Steps:
1️⃣ Apply for admission (take the PCAT entrance exam)
2️⃣ Wait for results — check the official portal
3️⃣ Submit required documents to the Registrar
4️⃣ Pay miscellaneous/other applicable fees
5️⃣ Confirm enrollment — get your class schedule
6️⃣ Check your subjects via the Student Portal

💡 Note: May difference ang process para sa Freshmen vs Transferees.

Sabihin mo lang:
• "Freshman ako" — para sa freshman guide
• "Transferee ako" — para sa transferee guide""",

"""📋 How to Enroll at PLMun

Ang enrollment process ay ganito:

1. PCAT — Kumuha ng application, mag-take ng entrance test
2. Results — Alamin kung pumasa ka
3. Documents — I-submit ang requirements sa Registrar
4. Payment — Bayaran ang applicable fees
5. Confirmation — I-lock ang enrollment
6. Portal — I-check ang schedule mo

👉 For freshmen: Need Form 138 + PSA + Good Moral + Medical
👉 For transferees: Need TOR + HD + PSA + Good Moral

Sino ka sa dalawa? 😊"""
],

"enrollment_freshmen": [
"""🎓 Freshman Enrollment Guide

Congrats sa plano mong pag-aral sa PLMun! 🎉

📌 Requirements:
• ✅ Form 138 (Grade 12 Report Card)
• ✅ PSA Birth Certificate
• ✅ Certificate of Good Moral Character
• ✅ Medical Certificate
• ✅ 2x2 ID Photos (2–4 copies)

📌 Steps:
1. Apply for PCAT (entrance exam)
2. Take the exam and wait for results
3. Submit the requirements to Registrar
4. Pay applicable fees
5. Get your class schedule

Good luck, future PLMunian! 💚""",

"""Welcome, Freshman! 😊🎓

Here's a quick guide to get you started:

📋 What You Need:
• Form 138 (Senior High Report Card)
• PSA Birth Certificate
• Good Moral Certificate
• Medical/Health Certificate
• ID photos

📋 What You'll Do:
1. Apply and take PCAT
2. Submit docs at Registrar
3. Pay fees → Confirm enrollment

📞 Questions? Call: 02-8248-9161
📧 plmuncomm@plmun.edu.ph"""
],

"enrollment_transferee": [
"""🔄 Transferee Enrollment Guide

Welcome to PLMun! 😊

📌 Requirements:
• ✅ Honorable Dismissal (from previous school)
• ✅ Transcript of Records (TOR)
• ✅ PSA Birth Certificate
• ✅ Certificate of Good Moral Character
• ✅ Medical Certificate

📌 Steps:
1. Take the PCAT entrance exam
2. Submit all requirements to Registrar
3. Undergo curriculum assessment
4. Pay applicable fees
5. Confirm enrollment

📞 Registrar: 02-8248-9161
📧 plmuncomm@plmun.edu.ph""",

"""So, transferee ka! Let me walk you through it 😊

📋 Kailangan Mo:
• Honorable Dismissal
• Transcript of Records
• PSA Birth Certificate  
• Good Moral Certificate
• Medical Certificate

📋 Proseso:
1. Take PCAT → Submit docs → Assessment → Pay → Done!

💡 Tip: Kumpletuhin mo lahat ng docs bago pumunta sa Registrar para hindi ka na babalik-balik 😊"""
],

"enrollment_pcat": [
"""📝 PCAT — PLMun College Admission Test

Ang PCAT ay ang entrance exam ng PLMun bago ka ma-enroll.

📌 Coverage:
• English (Reading + Grammar)
• Mathematics
• Science
• Filipino / Verbal Reasoning

📌 Important:
• Required sa lahat ng incoming freshmen at transferees
• Check official announcements for exam schedule
• Magdala ng valid ID at pencil sa exam day

💡 Study Tip: Focus on fundamentals — basic Math, English comprehension, and Science concepts!

📞 02-8248-9161 | 📧 plmuncomm@plmun.edu.ph""",

"""About the Entrance Exam (PCAT):

Ito ang gateway papasok ng PLMun! 📝

📌 Topics Covered:
• English
• Math
• Science
• Filipino

📌 Paano Mag-apply:
1. Pumunta sa Admissions Office
2. Fill out application form
3. Pay exam fee
4. Wait for your exam schedule

Mag-aral nang husto — kaya mo yan! 💪"""
],

"enrollment_tuition": [
"""💰 Tuition Fee Information

🎉 Good news! Ang PLMun ay covered ng RA 10931 (Free Tuition Law).

✅ Libre ang tuition para sa qualified students!

📌 Pero may iba pang fees:
• Miscellaneous fees
• Laboratory fees (depende sa course)
• Document fees

📌 Para sa exact breakdown:
👉 Pumunta sa Cashier / Treasury Office
👉 Or call: 02-8248-9161

💡 Tip: Always keep your official receipts! Importante yan for records.""",

"""About Tuition at PLMun 💰

PLMun ay state university kaya covered ng FREE tuition law! 🎉

✅ Tuition = FREE (under RA 10931)
❗ Other fees = may bayad pa rin (miscellaneous, lab fees, etc.)

📌 For exact amounts, contact:
• Cashier/Treasury Office
• 📞 02-8248-9161
• 📧 plmuncomm@plmun.edu.ph"""
],

"enrollment_online": [
"""💻 Online Enrollment

📌 PLMun may online enrollment process via Student Portal!

Steps:
1. Go to the Student Portal
2. Log in using your student ID + password
3. Check available subjects for the semester
4. Select your subjects / sections
5. Submit for approval
6. Wait for confirmation

💡 Tip: Mag-enroll agad — popular subjects fill up fast!

📞 If may issues: ICT Office / Registrar
🌐 portal.plmun.edu.ph"""
],

"enrollment_late": [
"""⏰ Late Enrollment

Nalampasan mo ang regular enrollment period? 😅

📌 What to do:
1. Pumunta sa Registrar's Office
2. Ask about late enrollment procedures
3. May additional requirements or fees

⚠️ Note: Late enrollment is not guaranteed — depende sa slots available.

📌 Contact:
📞 02-8248-9161
📧 plmuncomm@plmun.edu.ph

💡 Tip: Mag-enroll on time next sem! 😊"""
],

"enrollment_requirements": [
"""📋 Enrollment Requirements

Depende kung sino ka:

👤 FRESHMEN:
• Form 138 (Grade 12 Card)
• PSA Birth Certificate
• Good Moral Certificate
• Medical Certificate
• 2x2 Photos

🔄 TRANSFEREES:
• Honorable Dismissal
• Transcript of Records
• PSA Birth Certificate
• Good Moral Certificate
• Medical Certificate

📌 Submit all requirements sa Registrar's Office!"""
],

"enrollment_courses": [
"""📚 Courses Offered at PLMun

PLMun offers a range of bachelor's degree programs!

📌 Common Programs:
• BS Computer Science / Information Technology
• BS Business Administration
• BS Criminology
• BS Nursing / Health-related courses
• BA Communication
• BS Education
• And more!

📌 For the complete updated list:
👉 Visit www.plmun.edu.ph
👉 Or contact the Registrar / Admissions Office

📞 02-8248-9161"""
],

# ── SCHEDULE ────────────────────────────────────────────────────────────────

"schedule_general": [
"""📅 Class Schedule

Para makita ang class schedule mo:

1️⃣ Go to the Student Portal
2️⃣ Log in using your student ID + password
3️⃣ Click the "Schedule" or "Subjects" section

📌 You'll see:
• Subject names & codes
• Time and room assignments
• Your professor's name

💡 Tip: Screenshot your schedule! Laging updated ang portal.

🌐 portal.plmun.edu.ph""",

"""Your class schedule ay makikita sa Student Portal! 📅

📌 Steps:
1. Login sa portal (student ID + password)
2. Go to schedule section
3. Makikita mo na agad!

If may issue sa login → contact ICT Office.

📞 02-8248-9161"""
],

"schedule_check": [
"""🔍 How to Check Your Schedule

Super simple lang to! 😊

1. Open the PLMun Student Portal
2. Login using your student ID number as username
3. Default password: your birthdate (MMDDYYYY format)
4. Navigate to "My Schedule" or "Enrolled Subjects"

📌 Doon mo makikita:
• Time slots ng bawat subject
• Room assignments
• Days ng klase

💡 Wala ka pa ring account? Go to the Registrar or ICT Office!"""
],

"schedule_semester_start": [
"""📅 When Does the Semester Start?

📌 Typical Schedule:
• 1st Semester: Around August – September
• 2nd Semester: Around January – February

📌 Midyear / Summer: Usually May – June (if available)

⚠️ Note: Exact dates change every year! Always check:
• PLMun official website: www.plmun.edu.ph
• Official FB page: PLMun announcements
• Student Portal""",

"""Ah, pasukan na! 🎒

📌 Usually:
• 1st Sem: August or September
• 2nd Sem: January or February

📌 For exact dates, check:
• Academic Calendar sa official website
• Or call 📞 02-8248-9161

Exciting na ang bagong semester! 💚"""
],

"schedule_midterm": [
"""📚 Midterm Exam Schedule

📌 Midterm exams are usually held:
• Around the middle of the semester (6th–8th week)

📌 To find the exact dates:
• Check the Academic Calendar
• Ask your professor
• Visit the Student Portal for announcements

💡 Study Tips for Midterms:
• Start reviewing 1–2 weeks before
• Make a study schedule
• Don't cram! 😅

Good luck, you've got this! 💪"""
],

"schedule_finals": [
"""📚 Final Exam Schedule

Finals are around the corner! 😮

📌 Final exams are usually:
• Last 2 weeks of the semester
• Check exact dates via Student Portal or Academic Calendar

📌 Tips para hindi ka ma-stress:
• Review all topics from Week 1
• Make summary notes
• Get enough sleep before exam day
• Eat a good meal — brain food! 🧠

Kaya mo yan, future graduate! 💚 Good luck!""",

"""About Final Exams 📚

📌 Usually held: Last 2 weeks ng semester

📌 Where to see exact schedule:
• Student Portal
• Official announcements
• Check with your professor

Mag-aral nang mabuti! 💪 You got this! 💚"""
],

"schedule_break": [
"""🏖️ School Breaks & Vacation

📌 Major Breaks:
• Semestral Break: Between 1st and 2nd semester (around October–November)
• Christmas Vacation: December – January (exact dates vary)
• Holy Week: Usually 4–5 days

📌 For exact dates:
• Check the Academic Calendar on www.plmun.edu.ph
• Monitor official announcements

💡 Enjoy your break — but don't forget to prepare for the next semester! 😊"""
],

"schedule_suspension": [
"""☔ Class Suspension

📌 PLMun follows official class suspension orders from:
• CHED (Commission on Higher Education)
• Local Government (Muntinlupa City / Metro Manila)
• School administration announcements

📌 How to know if class is suspended:
• Check official PLMun Facebook page
• Monitor local news / PAGASA announcements
• Check your group chats and portal announcements

💡 Tip: Always wait for official PLMun announcements before assuming walang pasok! 😊"""
],

"schedule_dropping": [
"""📌 Adding/Dropping Period

The adding/dropping of subjects happens at the start of each semester!

📌 Timeline:
• Usually first 1–2 weeks of the semester

📌 Steps to Drop a Subject:
1. Get a dropping form from the Registrar
2. Have your professor sign it
3. Submit to Registrar before deadline

⚠️ Important: Dropping after the deadline = 5.00 grade (Failed)

📞 Registrar: 02-8248-9161"""
],

"schedule_calendar": [
"""📅 Academic Calendar

The Academic Calendar covers everything you need to know about the school year!

📌 Included:
• Enrollment dates
• Start of classes
• Midterm and Final exam weeks
• Holidays and school breaks
• Graduation ceremonies

📌 Where to get it:
• PLMun official website: www.plmun.edu.ph
• Student Portal
• Registrar's Office

💡 Save a copy on your phone — super useful throughout the year! 📱"""
],

# ── POLICIES ─────────────────────────────────────────────────────────────────

"policies_general": [
"""📚 PLMun Academic Policies

Here's a quick overview of important school policies:

📌 Grading:
• 1.00 = Excellent | 2.00 = Good | 3.00 = Passing | 5.00 = Failed

📌 Attendance:
• May maximum absences allowed per subject
• Exceeding limit → possible drop or failing grade

📌 Dropping:
• Allowed within the add/drop period
• Late dropping = 5.00

📌 Academic Integrity:
• Cheating / Plagiarism is strictly prohibited

📌 Residency:
• Maximum years to finish your course

👉 Sabihin mo kung alin gusto mong malaman in detail! 😊"""
],

"policies_grading": [
"""📊 PLMun Grading System

📌 Numerical Grade Equivalents:
• 1.00 — Excellent (97–100%)
• 1.25 — 94–96%
• 1.50 — 91–93%
• 1.75 — 88–90%
• 2.00 — 85–87% (Good)
• 2.25 — 82–84%
• 2.50 — 79–81%
• 2.75 — 76–78%
• 3.00 — 75% (Passing — minimum!)
• 5.00 — Below 75% (Failed)
• INC — Incomplete

💡 Target: 1.00! Kaya mo yan! 🌟""",

"""Grading System sa PLMun 📊

📌 Scale:
• 1.00 = Highest (Excellent)
• 3.00 = Passing grade
• 5.00 = Failed (below 75%)
• INC = Incomplete

📌 Tips para sa mataas na grade:
• Huwag pumasok nang huli
• Submit requirements on time
• Magtanong pag hindi naintindihan

Study hard! 💚"""
],

"policies_passing": [
"""✅ Passing Grade at PLMun

📌 The minimum passing grade is 3.00 (equivalent to 75%).

📌 What it means:
• Below 3.00 = you're good 🎉
• 5.00 = Failed — kailangan ulitin
• INC = May kulang na requirements — kailangan tapusin

📌 Tips:
• Always submit all requirements
• Attend classes regularly
• Communicate with your professor if may problema

Hindi ka babagsak kung mag-aaral ka nang husto! 💚"""
],

"policies_attendance": [
"""📋 Attendance Policy at PLMun

📌 General Rule:
• Each subject has a maximum number of allowed absences
• Usually: 20% of total class hours

📌 What happens if you exceed:
• Warning from professor
• Possible dropping from the subject
• Possible failing grade (5.00)

📌 Being Late:
• 3 late = 1 absence (typically)
• Depends on professor's policy

📌 Tips:
✅ Go to class regularly
✅ Inform your professor ahead of time if may valid reason
✅ Get a medical certificate if sick

Remember: Attendance matters! 💚""",

"""Attendance Rules sa PLMun 📋

📌 May allowed absences kada subject — pero limitado lang!

📌 Kapag sumobra:
• Possible drop
• Possible 5.00 (Fail)

📌 Late policy:
• 3 late = 1 absence (usually)

💡 Tip: Lagi kang dumating sa oras — grades mo ang affected! 😊

If may valid reason (sakit, emergency):
• Inform your professor ASAP
• Bring medical cert kung kailangan"""
],

"policies_attendance_allowed": [
"""✅ Allowed Absences

📌 Generally, you can be absent up to 20% of total class meetings.

Example:
• If subject meets 3x/week for 18 weeks = 54 classes
• 20% of 54 = ~10 absences maximum

📌 Note: This varies by subject and professor.

⚠️ Exceeding the limit means possible automatic drop or 5.00!

💡 Best advice: Huwag na lang mag-absent nang hindi kailangan 😊"""
],

"policies_attendance_importance": [
"""⭐ Why Attendance Matters at PLMun

📌 Attendance directly affects:
• Your final grade (class participation component)
• Whether you pass or fail the subject
• Your academic standing

📌 Beyond grades:
• You'll miss lectures and discussions
• Harder to catch up
• Professors notice consistent students 😊

📌 Bottom line:
Regular attendance = better grades + better learning! 💚"""
],

"policies_inc": [
"""📌 Incomplete (INC) Grade

Nakakuha ka ng INC? No worries! Here's how to resolve it:

📌 What is INC?
• Given when a student has missing requirements but has attended most classes

📌 How to Remove INC:
1. Contact the professor who gave the INC
2. Complete the missing requirements
3. Professor submits the updated grade
4. INC is replaced with your actual grade

⚠️ Important: INC has a deadline — usually within the next semester or as specified by the school.

📞 Registrar: 02-8248-9161"""
],

"policies_drop": [
"""📌 Dropping a Subject

Gusto mong mag-drop ng subject? Here's how:

📌 Steps:
1. Get a dropping/withdrawal form from the Registrar
2. Have your professor sign the form
3. Submit the signed form to the Registrar
4. Wait for confirmation

📌 Important Deadlines:
• Must be done within the official dropping period (usually first 2 weeks)
• Dropping after deadline = automatic 5.00 (Failed)

📌 Think before you drop:
• Will it affect your scholarship?
• Will it delay your graduation?
• Is there another option (e.g., talk to professor)?

📞 Registrar: 02-8248-9161"""
],

"policies_appeal": [
"""📋 Grade Appeal Process

Pakiramdam mo may mali sa grade mo? You can appeal! 😊

📌 Steps:
1. Talk to your professor first — ask for grade breakdown
2. If unresolved, go to the Department Chair
3. Submit a formal written appeal
4. Include evidence (graded papers, receipts, etc.)
5. Wait for the decision

📌 Tips:
• Be respectful and professional
• Keep all your graded papers and receipts
• Act quickly — there's usually a deadline for appeals

📞 02-8248-9161 | 📧 plmuncomm@plmun.edu.ph"""
],

"policies_shift": [
"""🔄 Shifting Courses

Nag-iisip kang mag-shift? Here's what you need to know:

📌 Steps:
1. Talk to your current department's Dean or adviser
2. Get a shifting form from the Registrar
3. Secure approval from the target department
4. Submit requirements (TOR, grades, etc.)
5. Await final approval

📌 Important:
• Some transferred units may not be credited
• Shifting may affect your graduation timeline
• Talk to your guidance counselor first!

📞 02-8248-9161
💬 Guidance Office: for counseling on shifting decisions"""
],

"policies_loa": [
"""📄 Leave of Absence (LOA)

May hindi inaasahang nangyari at kailangan mong huminto ng ilang panahon?

📌 Steps:
1. Fill out a LOA form (available at Registrar)
2. Get signatures: professor, adviser, dean
3. Submit to Registrar before deadline
4. Official confirmation is issued

📌 Important Notes:
• LOA is counted in your maximum residency years
• Hindi ito libre — may deadline at proseso
• Talk to guidance counselor before filing

📞 02-8248-9161 | 📧 plmuncomm@plmun.edu.ph"""
],

"policies_honors": [
"""🏅 Latin Honors & Dean's List

Gusto mong maging honor student? Here's how!

📌 Graduation Honors (Latin Honors):
• Summa Cum Laude: 1.00–1.20 GWA
• Magna Cum Laude: 1.21–1.45 GWA
• Cum Laude: 1.46–1.75 GWA

📌 Dean's List (per semester):
• Usually GWA of 1.75 or higher with no failing grades
• No INC grades
• Good standing (no disciplinary issues)

💡 Tip: Every subject matters — maging consistent ka! 💪

📞 Registrar for official criteria: 02-8248-9161"""
],

"policies_cheating": [
"""⚠️ Academic Integrity Policy

Cheating, plagiarism, and academic dishonesty are strictly prohibited at PLMun!

📌 What counts as cheating:
• Copying during exams
• Plagiarizing submitted work
• Using unauthorized materials
• Having someone else do your work

📌 Consequences:
• Automatic 5.00 (Fail) in the subject
• Disciplinary action
• Possible suspension or dismissal (for repeated offenses)

📌 Remember:
Your degree's value depends on your integrity. Study honestly! 💚"""
],

"policies_dressCode": [
"""👕 Dress Code / Uniform Policy

📌 General Rules:
• Proper school attire is required at all times
• Wearing of prescribed uniform (depends on course/department)
• Closed-toe shoes required
• Avoid overly revealing, torn, or inappropriate clothing

📌 PE Class:
• PE uniform required during Physical Education classes

📌 Laboratory:
• Lab gown or coat may be required depending on course

💡 Tip: When in doubt, dress conservatively and wear your uniform properly! 😊"""
],

"policies_retention": [
"""📌 Retention & Dismissal Policy

📌 Academic Probation:
• Placed on probation if GWA falls below minimum
• Given a semester to improve

📌 Academic Dismissal:
• If GWA is consistently failing
• Too many 5.00 grades
• Exceeding maximum residency period

📌 How to avoid it:
• Maintain at least 3.00 GWA
• Don't accumulate failing grades
• Seek guidance counselor help early!

📞 02-8248-9161"""
],

"policies_residency": [
"""⏳ Maximum Residency Rule

📌 What is it?
• Maximum number of years to complete your degree

📌 General Rule:
• Usually 1.5x the normal program length
• Example: 4-year course → maximum 6 years

📌 Important:
• LOA years are counted
• Shifting may add time
• Exceeding maximum residency → possible dismissal

📌 Check with Registrar for your specific program:
📞 02-8248-9161"""
],

# ── SECTION INFO ────────────────────────────────────────────────────────────

"section_info": [
"""📘 Your Class Section

Para malaman ang section mo:

1️⃣ Login sa Student Portal
2️⃣ Go to "My Profile" or "Enrollment Details"
3️⃣ Makikita mo ang iyong section, block, at subjects

📌 If hindi pa available:
• Contact Registrar's Office
• Wait for official posting

📞 02-8248-9161
📧 plmuncomm@plmun.edu.ph""",

"""Section Information 📘

Ang section mo ay nasa Student Portal! Here's how:

1. Login sa portal
2. Check "Enrollment Details" or "My Subjects"
3. Makikita ang section/block assignment

Still wala? → Call Registrar: 02-8248-9161"""
],

# ── DOCUMENTS ───────────────────────────────────────────────────────────────

"documents_general": [
"""📄 Document Requests

Anong document ang kailangan mo?

📌 Available from Registrar's Office:
• 📜 Transcript of Records (TOR)
• 📋 Certificate of Enrollment (COE)
• 🏅 Good Moral Character Certificate
• 🎓 Diploma (replacement)
• ✅ Clearance
• 📁 Form 137
• 🔓 Honorable Dismissal

📌 General Steps:
1. Go to Registrar's Office
2. Fill out request form
3. Pay the applicable fee at the Cashier
4. Submit receipt and wait for processing

📌 Processing time: 1–5 business days depending on document.

📞 02-8248-9161
📧 plmuncomm@plmun.edu.ph""",

"""Need a document? I can help! 📄

📌 Common Requests:
• TOR → 3–5 days
• COE → 1–3 days
• Good Moral → 1–3 days
• Diploma → 5+ days

📌 Steps: Go to Registrar → Fill form → Pay → Wait

Sabihin mo lang kung anong specific na document! 😊"""
],

"documents_tor": [
"""📄 Transcript of Records (TOR)

📌 Steps to Request:
1. Go to the Registrar's Office
2. Fill out a TOR request form
3. Pay the corresponding fee at the Cashier
4. Submit receipt back to Registrar
5. Wait for processing (3–5 business days)
6. Claim with valid ID

📌 Important:
• Must have no pending obligations (clearance)
• May take longer if many requests are pending

📌 Rush / Special requests:
• Ask Registrar for rush processing options

📞 02-8248-9161""",

"""How to Get Your TOR 📄

1. Pumunta sa Registrar's Office
2. Fill out request form
3. Pay sa Cashier
4. Wait 3–5 days
5. Claim with your valid ID

💡 Tip: Make sure wala kang outstanding obligations bago mag-request!

📞 02-8248-9161"""
],

"documents_coe": [
"""📋 Certificate of Enrollment (COE)

📌 Steps:
1. Go to Registrar's Office
2. Fill out COE request form
3. Pay processing fee at Cashier
4. Wait 1–3 business days
5. Claim with your student ID

📌 Uses of COE:
• Proof of enrollment for scholarships
• PWD/senior benefits
• Bank requirements
• Employer proof

📞 02-8248-9161"""
],

"documents_goodmoral": [
"""🏅 Good Moral Certificate

📌 Steps to Request:
1. Go to Registrar's Office
2. Fill out request form
3. Pay processing fee at Cashier
4. Wait 1–3 business days
5. Claim with valid ID

📌 Note:
• May require endorsement from Guidance Office
• Usually required for: scholarships, applications, board exams

📞 02-8248-9161"""
],

"documents_diploma": [
"""🎓 Diploma Replacement

Nawala ang diploma mo? Here's what to do:

📌 Steps:
1. Go to Registrar's Office
2. Request for diploma replacement
3. Submit valid ID + affidavit of loss (if required)
4. Pay replacement fee
5. Wait for processing (may take several days/weeks)

📌 Note:
• May require clearance first
• Processing time is longer than other documents

📞 02-8248-9161
📧 plmuncomm@plmun.edu.ph"""
],

"documents_clearance": [
"""✅ School Clearance

📌 What is it?
• Certification that you have no outstanding obligations with the school

📌 Steps:
1. Get clearance form from Registrar
2. Get signatures from all required offices (library, cashier, etc.)
3. Submit completed form to Registrar

📌 Required for:
• TOR request
• Honorable Dismissal
• Diploma
• Graduation

📞 02-8248-9161"""
],

"documents_form137": [
"""📁 Form 137

📌 Form 137 is your Permanent Record from senior high school.

📌 For PLMun purposes:
• Usually requested from your previous school (SHS)
• Submit to PLMun Registrar during enrollment

📌 If requesting from PLMun:
• Go to Registrar's Office
• Fill out request form
• Pay applicable fee

📞 02-8248-9161"""
],

"documents_hd": [
"""🔓 Honorable Dismissal

📌 What is it?
• A document certifying that you left the school in good standing

📌 Required for:
• Transferring to another school

📌 Steps:
1. Settle all obligations (clearance)
2. Go to Registrar
3. Fill out HD request form
4. Pay processing fee
5. Wait and claim

📞 02-8248-9161"""
],

"documents_boardexam": [
"""📋 Board Exam / PRC Application Documents

📌 Usually Required:
• Transcript of Records (TOR)
• Certificate of Good Moral
• Diploma

📌 Steps:
1. Request documents from Registrar
2. Get certified/authenticated copies if needed
3. Submit to PRC based on their requirements

📌 Note:
• PRC requirements may vary per exam — always check PRC website
• 🌐 www.prc.gov.ph

📞 Registrar: 02-8248-9161"""
],

"documents_proxy": [
"""👥 Document Pickup by Proxy

Hindi mo kayang personal na i-pick up ang documents mo?

📌 Steps:
1. Prepare an Authorization Letter (signed by you)
2. The authorized person must bring:
   • Your Authorization Letter
   • Your valid ID (photocopy)
   • Their own valid ID (original)

📌 Note:
• Some documents require personal claim only
• Verify with Registrar which documents allow proxy claiming

📞 02-8248-9161"""
],

"documents_fee": [
"""💰 Document Fees

📌 Document fees vary depending on the type of document.

📌 General range:
• COE: ~₱50–₱100
• Good Moral: ~₱50–₱100
• TOR: ~₱200–₱500 (depends on number of copies)
• Diploma replacement: Higher fee

📌 For exact amounts:
• Go to Cashier/Treasury Office
• Or call: 📞 02-8248-9161

Always pay official fees and get a receipt! 🧾"""
],

# ── SERVICES ─────────────────────────────────────────────────────────────────

"services_general": [
"""🏫 PLMun Student Services

Maraming available na services para sa inyo! 😊

📌 Available:
• 🎓 Scholarships (PLMun ng Bayan + others)
• 🪪 Student ID (new / replacement)
• 💻 Student Portal access
• 💬 Guidance & Counseling
• 📚 Library (e-library + physical)
• 🏥 School Clinic
• 💰 Cashier / Treasury
• 👥 Student Organizations

👉 Sabihin mo kung anong service ang kailangan mo — mabilis ako! 😊"""
],

"services_PLmunnbayan": [
"""🎓 PLMun ng Bayan Scholarship

Ito ang city scholarship program ng Muntinlupa City para sa mga PLMun students!

📌 Who can apply:
• Muntinlupa City residents
• Qualified based on financial need and academic standing

📌 How to apply:
1. Check announcements from PLMun / City Government
2. Prepare requirements (residency proof, grades, etc.)
3. Submit at designated office

📌 Benefits:
• Financial assistance / allowance
• Educational support

📌 Contact:
• Student Affairs Office
• 📞 02-8248-9161
• Check PLMun official announcements"""
],

"services_scholarship": [
"""🎓 Scholarships at PLMun

Great news — may iba't ibang scholarship options! 🎉

📌 Available Types:
1. PLMun ng Bayan — for Muntinlupa residents
2. Academic Excellence — for high-achieving students
3. Government Scholarships (CHED, etc.)
4. External / Private scholarships

📌 General Requirements:
• Good academic standing (usually 2.0 or higher GWA)
• No failing grades / INC
• Financial need (for need-based)
• Residency requirements (for city scholarships)

📌 How to Apply:
1. Go to Student Affairs Office (OSA)
2. Check requirements and forms
3. Submit complete documents
4. Wait for evaluation

📞 02-8248-9161 | 📧 plmuncomm@plmun.edu.ph""",

"""About Scholarships 🎓

Mag-apply ka! Baka may pwesto para sa iyo 😊

📌 Types:
• PLMun ng Bayan (city scholarship)
• Academic scholarships
• Government-funded (CHED, etc.)

📌 Apply sa:
• Student Affairs / OSA office

📞 02-8248-9161"""
],

"services_id": [
"""🪪 Student ID

📌 New Student ID:
1. Go to the Office of Student Affairs (OSA)
2. Submit required photo + documents
3. Pay ID fee
4. Wait for ID release

📌 Lost ID Replacement:
1. Execute Affidavit of Loss (have it notarized)
2. Submit affidavit + valid ID to OSA
3. Pay replacement fee
4. Wait for new ID

📌 Always carry your student ID:
• Required for library, clinic, and most school services

📞 02-8248-9161""",

"""Need a student ID? Here's how 🪪

New ID:
• Go to OSA → Submit photo → Pay → Wait

Lost ID:
• Notarized Affidavit of Loss → OSA → Pay → Wait

Simple lang! 😊 📞 02-8248-9161"""
],

"services_portal": [
"""💻 PLMun Student Portal

📌 URL: portal.plmun.edu.ph

📌 How to Login:
• Username: Student ID Number
• Password: Birthdate in MMDDYYYY format (default)

📌 Common Issues:
• Forgot password → Contact ICT Office
• Can't login → Make sure you have an active enrollment

📌 What you can do in the portal:
• View class schedule
• Check grades
• Monitor enrollment status
• View announcements

📞 ICT Office: 02-8248-9161
📧 plmuncomm@plmun.edu.ph""",

"""Student Portal Help 💻

📌 Login at: portal.plmun.edu.ph
• User: Your student ID
• Password: MMDDYYYY (birthdate)

📌 Can't log in?
→ Go to ICT Office or contact:
📞 02-8248-9161"""
],

"services_guidance": [
"""💬 Guidance & Counseling Office

You're not alone! The Guidance Office is here for you 💚

📌 Services:
• Individual counseling sessions
• Group counseling
• Career advising
• Academic advising
• Mental health support

📌 Why go?
• Feeling stressed about school?
• Having personal/family problems?
• Career or course uncertainty?
• Just need someone to talk to?

📌 It's confidential! 🔒

📌 Location: Check campus map or ask any admin office

📞 02-8248-9161"""
],

"services_library": [
"""📚 PLMun Library

📌 Services:
• Borrow physical books and materials
• Use the e-library for digital resources
• Study area / reading rooms
• Print and photocopy services

📌 Rules:
• Bring your student ID to borrow books
• Respect quiet zones
• Return books on time — fines may apply

📌 E-Library Access:
• May online access via student portal or library system

📞 02-8248-9161"""
],

"services_clinic": [
"""🏥 School Clinic / Health Services

📌 Services Available:
• First aid treatment
• Basic medical check-up
• Medicine for common illnesses
• Medical certificates (may be required for absences)
• Dental services (may be scheduled)

📌 Location:
• Inside the campus — ask the guard or admin office for exact location

📌 Tip:
• Always bring your student ID when visiting
• Medical certificate from clinic can support your excuse for absences

📞 02-8248-9161"""
],

"services_cashier": [
"""💰 Cashier / Treasury Office

📌 Payments accepted:
• Miscellaneous fees
• Document request fees
• ID replacement fees
• Other school-related fees

📌 Important:
• Always get an official receipt! 🧾
• Keep receipts for your records

📌 Location:
• Inside PLMun campus — ask at the front desk or guard

📞 02-8248-9161"""
],

"services_contact": [
"""📞 Contact PLMun

Here are all the ways to reach PLMun:

📞 Phone: 02-8248-9161
📧 Email: plmuncomm@plmun.edu.ph
🌐 Website: www.plmun.edu.ph

📌 Offices:
• Registrar — enrollment, documents, records
• Cashier — payments and fees
• OSA — student affairs, ID, orgs
• ICT Office — portal and technical issues
• Guidance — counseling and advising

🕐 Office hours: Mon–Fri, 8AM–5PM (check for updates)""",

"""PLMun Contact Info 📞

📞 02-8248-9161
📧 plmuncomm@plmun.edu.ph
🌐 www.plmun.edu.ph

Mon–Fri, regular office hours! 😊"""
],

"services_location": [
"""📍 PLMun Location

📌 Address:
Pamantasan ng Lungsod ng Muntinlupa
Poblacion, Muntinlupa City
Metro Manila, Philippines

📌 How to Get There:
• Jeepney / Bus: Routes passing Muntinlupa Poblacion
• UV Express: Available from nearby terminals
• Grab / Taxi: Just type the school name in the app

📌 Near landmarks:
• Close to Muntinlupa City Hall area

🌐 Check Google Maps: "PLMun" or "Pamantasan ng Lungsod ng Muntinlupa" """,

"""PLMun is located in Muntinlupa City! 📍

📌 Poblacion, Muntinlupa City, Metro Manila

📌 How to go:
• Jeep / Bus → Muntinlupa area
• Grab or taxi → Search "PLMun"

🌐 www.plmun.edu.ph"""
],

"services_org": [
"""👥 Student Organizations

PLMun has active student organizations and clubs!

📌 Types:
• Academic organizations (department-based)
• Special interest clubs (arts, sports, music, etc.)
• Student Government (SSG)

📌 How to Join:
1. Ask at the Office of Student Affairs (OSA)
2. Attend org orientations / General Assembly
3. Fill out membership form
4. Participate in org activities!

📌 Why join?
• Leadership development
• Build your network
• Extracurricular achievements for resume

📞 OSA: 02-8248-9161"""
],

"services_complaint": [
"""⚠️ Filing a Complaint or Concern

Mayroon kang reklamo o concern? You deserve to be heard! 💚

📌 Steps:
1. Document your complaint (what happened, when, who was involved)
2. Go to:
   • Guidance Office (personal/interpersonal concerns)
   • Student Affairs Office / OSA (general student issues)
   • Department Chair (academic-related)
3. File a written complaint form
4. Follow up for resolution

📌 Note:
• Confidential complaints are respected
• No retaliation for valid concerns

📞 02-8248-9161
📧 plmuncomm@plmun.edu.ph"""
],
}

# =============================================================================
# SMART FALLBACK
# =============================================================================

TOPIC_SUGGESTIONS = {
    "enrollment": [
        "How to enroll as a freshman or transferee",
        "What documents do I need for enrollment?",
        "How to apply for the PCAT entrance exam",
        "What courses does PLMun offer?",
        "Is there online enrollment available?",
    ],
    "schedule": [
        "How to check my class schedule",
        "When is the midterm or final exam?",
        "When does the adding/dropping period start?",
        "What is the academic calendar for this semester?",
        "Are there class suspensions I should know about?",
    ],
    "policies": [
        "How does the PLMun grading system work?",
        "What is the passing grade at PLMun?",
        "How many absences are allowed per subject?",
        "How do I remove an INC grade?",
        "Can I shift to a different course?",
    ],
    "documents": [
        "How to request a Transcript of Records (TOR)",
        "How to get a Certificate of Enrollment",
        "How to request a Good Moral Certificate",
        "What documents do I need for board exam application?",
        "Can someone else claim my documents for me?",
    ],
    "services": [
        "How to apply for the PLmun ng Bayan scholarship",
        "How to get or replace my student ID",
        "I can't log in to the student portal — what do I do?",
        "Where is the PLMun campus located?",
        "How to reach the Guidance and Counseling Office?",
    ],
}

def get_smart_fallback(user_input):
    cleaned = user_input.lower()
    detected_area = None
    area_keywords = {
        "enrollment": ["enroll", "admission", "apply", "pcat", "course", "program", "tuition", "fee"],
        "schedule": ["schedule", "exam", "midterm", "finals", "calendar", "class", "time", "when"],
        "policies": ["grade", "absent", "drop", "policy", "rule", "inc", "shift", "loa", "honor"],
        "documents": ["tor", "certificate", "document", "diploma", "clearance", "form", "record"],
        "services": ["scholarship", "id", "portal", "library", "clinic", "guidance", "cashier", "contact"],
    }
    for area, keywords in area_keywords.items():
        if any(kw in cleaned for kw in keywords):
            detected_area = area
            break

    if detected_area:
        suggestions = random.sample(TOPIC_SUGGESTIONS[detected_area], min(3, len(TOPIC_SUGGESTIONS[detected_area])))
        intro = f"Hmm, I'm not quite sure about that specific **{detected_area.capitalize()}** question, but here are some related things I can help with:"
    else:
        areas = random.sample(list(TOPIC_SUGGESTIONS.keys()), 3)
        suggestions = [random.choice(TOPIC_SUGGESTIONS[a]) for a in areas]
        intro = "Hmm, I'm not sure I caught that one! 😅 Here are some things I can help you with:"

    suggestion_lines = "\n".join([f"• {s}" for s in suggestions])
    openers = [
        "Hmm, that one's a bit outside what I know right now! 😊",
        "Oops, I don't have a direct answer for that yet — sorry!",
        "That's a tough one for me! Let me suggest some related topics:",
        "I'm not quite sure about that one, but here are similar things I know about:",
    ]

    return f"""{random.choice(openers)}

{intro}

{suggestion_lines}

You can also try rephrasing your question, or reach out to PLMun directly:
📞 02-8248-9161 | 📧 plmuncomm@plmun.edu.ph | 🌐 www.plmun.edu.ph

I'm always here to help — just ask away! 💚"""


# =============================================================================
# GET RESPONSE — with variation + rephrasing
# =============================================================================

def get_response(intent, user_input="", context=None):
    """
    Returns a full response string, with:
    - Smart rephrasing prefix (if not greeting/farewell/thanks)
    - Randomized response from multiple variants
    - Context-aware follow-up handling
    """
    if intent == "unknown":
        return get_smart_fallback(user_input)

    # Get response body
    response_data = RESPONSES.get(intent)
    if response_data is None:
        return get_smart_fallback(user_input)

    # Pick random variant if list
    if isinstance(response_data, list):
        body = random.choice(response_data)
    else:
        body = response_data

    # Skip rephrasing for conversational intents
    skip_rephrase = intent in ("greeting", "farewell", "thanks", "unknown")

    if skip_rephrase:
        return body

    # Add rephrasing prefix
    rephrase = get_rephrase(intent, user_input)
    return f"{rephrase}\n\n{body}"


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def train_model(texts, labels):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ('nb', MultinomialNB(alpha=0.3))
    ])
    pipeline.fit([preprocess(t) for t in texts], labels)
    return pipeline

def evaluate_model(texts, labels, model_pipeline):
    processed = [preprocess(t) for t in texts]
    label_counts = Counter(labels)
    min_count = min(label_counts.values())

    if min_count < 2:
        X_train, X_test, y_train, y_test = train_test_split(processed, labels, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(processed, labels, test_size=0.2, random_state=42, stratify=labels)

    ep = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ('nb', MultinomialNB(alpha=0.3))
    ])
    ep.fit(X_train, y_train)
    y_pred = ep.predict(X_test)
    intents = sorted(set(labels))

    try:
        cv = cross_val_score(ep, processed, labels, cv=5, scoring='accuracy')
        cv_scores = [round(s * 100, 2) for s in cv]
        cv_mean = round(cv.mean() * 100, 2)
        cv_std = round(cv.std() * 100, 2)
    except:
        cv_scores, cv_mean, cv_std = [], 0, 0

    report = classification_report(y_test, y_pred, labels=intents, target_names=intents, output_dict=True, zero_division=0)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        "cross_val_mean": cv_mean,
        "cross_val_std": cv_std,
        "cv_scores": cv_scores,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_samples": len(labels),
        "intents": intents,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=intents).tolist(),
        "report": report,
    }


# =============================================================================
# STARTUP — Load, Train, Evaluate
# =============================================================================

print("Loading dataset...", end=" ")
texts, labels = load_dataset(DATASET_FILE)
print(f"Done. ({len(texts)} samples)")
print("Training model...", end=" ")
pipeline = train_model(texts, labels)
print("Done.")
eval_results = evaluate_model(texts, labels, pipeline)
print(f"Accuracy: {eval_results['accuracy']}% | F1: {eval_results['f1_score']}% | CV: {eval_results['cross_val_mean']}%")
print("PLMun Chatbot v2.0 Ready! 💚")


# =============================================================================
# FLASK ROUTES
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

    # Get session context
    context = get_context()

    # Classify
    intent, method, confidence = smart_classify(user_input, context)

    # Refine sub-intent
    if intent != "unknown":
        intent = refine_intent(intent, user_input)

    # Update context memory
    update_context(intent)

    response = get_response(intent, user_input, context)

    return jsonify({
        "response": response,
        "intent": intent,
        "method": method,
        "confidence": confidence,
        "response_time_ms": round((time.time() - start) * 1000, 2),
    })

@app.route("/reset", methods=["POST"])
def reset_context():
    """Clear conversation context/memory."""
    session.clear()
    return jsonify({"status": "context cleared"})

@app.route("/eval")
def get_eval():
    return jsonify(eval_results)

@app.route("/stats")
def get_stats():
    return jsonify({
        "total_samples": len(texts),
        "total_intents": len(set(labels)),
        "intents": sorted(set(labels)),
        "total_specific_responses": len(RESPONSES),
        "algorithm": "Hybrid v2: Rule-Based + Context Memory + Fuzzy + TF-IDF + Multinomial Naive Bayes",
        "accuracy": eval_results["accuracy"],
        "f1_score": eval_results["f1_score"],
        "features": ["Context Memory", "Intent Refinement", "Response Variation", "Smart Rephrasing", "Taglish Normalization"],
    })

if __name__ == "__main__":
    app.run(debug=True)