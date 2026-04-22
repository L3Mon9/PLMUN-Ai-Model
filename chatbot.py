# =============================================================================
# FLASK WEB APP — AI Institutional Support Chatbot
# PLMun — Pamantasan ng Lungsod ng Muntinlupa
# University Road, NBP Reservation, Brgy. Poblacion, Muntinlupa City, 1776
# =============================================================================

import re, csv, os, random, time
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
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

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
        "enroll late", "online enrollment", "enrollment fee", "add subjects",
        "courses offered", "course available", "what course", "installment",
        "miscellaneous fee", "form 138", "placement test", "pcat",
        "college admission test", "regular student", "irregular student",
        "cross enroll", "quota", "freshmen", "freshman", "new student"
    ],
    "schedule": [
        "schedule", "class", "classes", "semester", "academic calendar",
        "timetable", "calendar", "holiday", "semestral break",
        "midterm", "final exam", "first day of class", "make up class",
        "class suspended", "last day of class", "school day",
        "kelan ang", "kailan magsisimula", "may pasok", "anong oras",
        "dropping period", "summer class", "exam week", "recognition day",
        "graduation", "school year", "christmas vacation",
        "second semester", "first semester", "academic year", "college week"
    ],
    "policies": [
        "policy", "policies", "rules", "grade", "grading", "attendance",
        "absent", "absences", "fail", "failing", "shift", "dropout",
        "dress code", "retention", "conduct", "appeal", "incomplete",
        "cheating", "misconduct", "leave of absence", "residency",
        "passing grade", "drop subject", "bagsak", "bumagsak",
        "pano kung bumagsak", "ilan ang allowed", "probation",
        "grading scale", "final grade", "academic probation",
        "latin honor", "cum laude", "magna cum laude", "summa cum laude",
        "deans list", "dismissed", "dismissal", "expulsion", "suspension",
        "tardiness", "plagiarism", "inc grade", "conditional grade",
        "maximum residency", "gadgets", "uniform", "code of conduct",
        "maximum absence", "retake", "remove grade"
    ],
    "documents": [
        "tor", "transcript", "certificate", "registrar", "document",
        "diploma", "clearance", "form 137", "credentials", "release",
        "authorization", "good moral", "claim", "honorable dismissal",
        "proof of enrollment", "copy of grades", "paano kumuha",
        "saan kukuha", "bayad sa dokumento", "official record",
        "board exam", "civil service", "authentication", "certified copy",
        "diploma replacement", "lost diploma", "follow up", "rush processing",
        "certificate of graduation", "units earned", "medium of instruction",
        "affidavit of loss"
    ],
    "services": [
        "service", "services", "guidance", "scholarship", "financial aid",
        "support", "portal", "clinic", "library", "dormitory",
        "cashier", "canteen", "organization", "lost and found",
        "counseling", "school nurse", "student org", "financial help",
        "lost id", "pay tuition", "mag apply ng scholarship",
        "nasaan ang guidance", "saan ang cashier", "may scholarship",
        "helpdesk", "hotline", "complaint", "student affairs",
        "student id", "replace id", "id replacement",
        "wifi", "internet", "computer lab", "printing", "photocopy",
        "campus", "mental health", "facebook", "website", "email",
        "contact number", "office hours", "student council",
        "join club", "student organization", "iskolar ng bayan",
        "CHED scholarship", "indigency", "tuition waiver", "financial assistance",
        "reset password", "cannot login", "student portal problem",
        "eLibrary", "elibrary", "library card"
    ]
}

def rule_based_classify(text):
    for intent, keywords in KEYWORD_MAP.items():
        for keyword in keywords:
            if keyword in text:
                return intent
    return None

# =============================================================================
# RESPONSES — Based on Real PLMun Information
# =============================================================================
RESPONSES = {

    "greeting": [
        """👋 Hello, Iskolar! Welcome to the PLMun Student Support Chatbot!

I am your official AI-powered assistant for Pamantasan ng Lungsod ng Muntinlupa (PLMun) — located at University Road, NBP Reservation, Brgy. Poblacion, Muntinlupa City.

📌 I can help you with:
• 📋 Enrollment & Admission — PCAT, requirements, procedures
• 📅 Class Schedule — academic calendar, exam dates, semestral breaks
• 📜 School Policies — grading system, attendance, conduct
• 📁 Documents — TOR, certificates, diplomas, clearances
• 🎓 Student Services — Iskolar ng Bayan scholarship, guidance, library, clinic

💚 PLMun Color: Bamboo Green | Motto: Serve the City, Serve the Nation

How can I help you today? 😊""",

        """🎓 Mabuhay, Iskolar ng PLMun!

I'm your Student Support Chatbot for Pamantasan ng Lungsod ng Muntinlupa. Whether you have questions about PCAT admission, enrollment, your schedule, policies, documents, or scholarships — I'm here to guide you step by step!

📞 PLMun Contact:
• Telephone: 02-8248-9161 loc. 146
• Email: plmuncomm@plmun.edu.ph
• Website: www.plmun.edu.ph

What would you like to know today? 😊""",
    ],

    "farewell": [
        """👋 Thank you for using the PLMun Student Support Chatbot!

Before you go, here are quick reminders:
✅ Check the PLMun website (www.plmun.edu.ph) for latest announcements.
✅ Monitor your Student Portal regularly for grades and enrollment updates.
✅ For urgent concerns, visit the Registrar's Office or Student Affairs Office during office hours (Mon–Fri, 8:00 AM – 5:00 PM).

💚 Good luck with your studies, Iskolar! Serve the City, Serve the Nation! 💪
Feel free to come back anytime. Goodbye! 😊""",

        """See you, Iskolar! 👋

Remember — PLMun is always here to support you.
📞 02-8248-9161 | 🌐 www.plmun.edu.ph | 📧 plmuncomm@plmun.edu.ph

Take care and keep pushing forward! 💚""",
    ],

    "thanks": [
        """😊 You're very welcome, Iskolar!

Here's what you can do next:
✅ If your concern is resolved — go ahead and take the next steps!
✅ If you need to visit an office — bring your Student ID and required documents.
✅ For more info — visit www.plmun.edu.ph or call 02-8248-9161.

Is there anything else I can help you with? 🎓""",

        """Glad I could help! 😊

Feel free to come back if you have more questions about PLMun enrollment, schedules, policies, documents, or services. Good luck, Iskolar! 💚""",
    ],

    "enrollment": [
        """📋 PLMun ADMISSION & ENROLLMENT GUIDE

Welcome to Pamantasan ng Lungsod ng Muntinlupa (PLMun)! Here is a complete guide on how to get admitted and enrolled.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 1 — TAKE THE PLMun COLLEGE ADMISSION TEST (PCAT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• PLMun requires applicants to pass the PLMun College Admission Test (PCAT) before enrollment.
• Apply online at: https://plmun.edu.ph/admission/undergrad/
• Fill out the online application form completely.
• Wait for your examination schedule and venue.

📝 PCAT Requirements:
✅ Online Application Form (filled out completely)
✅ Form 138 / Senior High School Report Card (photocopy)
✅ PSA Birth Certificate (photocopy)
✅ Certificate of Good Moral Character (from your school)
✅ 2x2 ID photo (white background)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 2 — PREPARE YOUR ENROLLMENT DOCUMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After passing the PCAT, prepare the following:

For NEW STUDENTS (Freshmen / SHS Graduates):
✅ Original Form 138 (Senior High School Report Card)
✅ PSA Birth Certificate (original + 1 photocopy)
✅ Certificate of Good Moral Character (original)
✅ Medical Certificate from a licensed physician
✅ 2x2 ID photos (2 copies, white background)
✅ PCAT Result / Notice of Acceptance

For TRANSFEREES:
✅ Honorable Dismissal from previous school
✅ Certified True Copy of Transcript of Records (TOR)
✅ PSA Birth Certificate (original + 1 photocopy)
✅ Certificate of Good Moral Character
✅ Medical Certificate
✅ 2x2 ID photos (2 copies)

For OLD / CONTINUING STUDENTS:
✅ Valid PLMun Student ID
✅ Previous Certificate of Registration (COR)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 3 — GO TO THE REGISTRAR'S OFFICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Visit the PLMun Registrar's Office during the official enrollment period.
• Email: universityregistrar@plmun.edu.ph
• Submit all required documents for verification.
• Get your Enrollment Assessment Form.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 4 — PAY AT THE TREASURY / CASHIER'S OFFICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• PLMun offers FREE TUITION under the UNIFAST / Universal Access to Quality Tertiary Education Act (RA 10931).
• Miscellaneous fees may still apply — ask the Cashier's Office for the current fee schedule.
• Payment can be made at the PLMun Treasury Office.
• Online payment options: Land Bank of the Philippines or Development Bank of the Philippines (mobile banking).
• Keep your Official Receipt — this is required to complete enrollment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 5 — CONFIRM YOUR ENROLLMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Return to the Registrar's Office with your Official Receipt.
• Receive your Certificate of Registration (COR).
• Check your Student Portal to confirm your enrolled subjects.

📞 For more info: 02-8248-9161 loc. 146 | 🌐 www.plmun.edu.ph
⚠️ Always enroll within the official period to avoid late enrollment penalties.

Do you have more questions about enrollment? 😊""",

        """📋 PLMun COURSES OFFERED

Pamantasan ng Lungsod ng Muntinlupa (PLMun) offers the following degree programs:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🖥️ COLLEGE OF INFORMATION TECHNOLOGY & COMPUTER STUDIES (CITCS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Bachelor of Science in Computer Science (BSCS)
✅ Bachelor of Science in Information Technology (BSIT)
✅ Associate in Computer Technology (ACT)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💼 COLLEGE OF BUSINESS ADMINISTRATION (CBA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ BS Business Administration — Major in Human Resource Development Management
✅ BS Business Administration — Major in Marketing Management
✅ BS Business Administration — Major in Operations Management

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 COLLEGE OF ACCOUNTANCY (COA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Bachelor of Science in Accountancy (BSA)
✅ Bachelor of Science in Management Accounting (BSMA)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 COLLEGE OF ARTS AND SCIENCES (CAS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Bachelor of Arts in Communication
✅ Bachelor of Science in Psychology
✅ Bachelor of Public Administration
✅ Bachelor of Arts in Political Science

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚔 COLLEGE OF CRIMINAL JUSTICE (CCJ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Bachelor of Science in Criminology
✅ Bachelor of Science in Industrial Security Management

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 COLLEGE OF TEACHER EDUCATION (CTE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Bachelor of Elementary Education (BEEd) — General Elementary Education
✅ Bachelor of Secondary Education (BSEd) — Major in Science
✅ Bachelor of Secondary Education (BSEd) — Major in English
✅ Bachelor of Secondary Education (BSEd) — Major in Social Science

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏥 COLLEGE OF MEDICINE (COM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Doctor of Medicine (MD)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤝 INSTITUTE OF SOCIAL WORK (ISW)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Bachelor of Science in Social Work (BSSW)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎓 GRADUATE STUDIES (GS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Master in Business Administration (MBA)
✅ Master of Arts in Education — Major in Educational Management
✅ Master of Arts in Education — Major in Guidance and Counseling
✅ Master in Security and Correctional Administration
✅ Master in Information Technology
✅ Master of Science in Criminology

🌐 For more details: www.plmun.edu.ph/program-offered.php
Do you want to know the admission requirements for a specific course? 😊""",
    ],

    "schedule": [
        """📅 CLASS SCHEDULE GUIDE — PLMun

Here is how to find and manage your class schedule at PLMun:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO CHECK YOUR SCHEDULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Go to the PLMun Student Portal: https://plmun.edu.ph/student-portal/system/main/index.php
Step 2 → Log in using your PLMun student credentials (Student ID + Password).
Step 3 → Navigate to the Schedule or Enrollment section.
Step 4 → View your list of subjects, class times, and room assignments.
Step 5 → Note down or print your schedule for reference.

If you can't access the Student Portal:
→ Visit the Registrar's Office (Mon–Fri, 8:00 AM – 5:00 PM)
→ Email: universityregistrar@plmun.edu.ph
→ Bring your Student ID for verification.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 PLMun ACADEMIC CALENDAR OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLMun follows a semester-based academic year:

📘 1st Semester — August to December
📗 2nd Semester — January to May
☀️ Summer Term — June to July (selected subjects only)

Key events in the calendar:
✅ Enrollment Period (before each semester starts)
✅ First Day of Classes
✅ Adding / Dropping Period (first 2 weeks of the semester)
✅ Midterm Examination Week (around Week 9)
✅ Final Examination Week (last 2 weeks of semester)
✅ Semestral Break (between semesters)
✅ Christmas Vacation (December)
✅ Recognition Day / Graduation Ceremonies (end of school year)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO REQUEST A SCHEDULE CHANGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Talk to your Academic Adviser about your concern.
Step 2 → Get a Schedule Change / Adding-Dropping Form from the Registrar's Office.
Step 3 → Fill out the form and have it signed by your professor and adviser.
Step 4 → Submit the approved form to the Registrar's Office before the dropping deadline.
Step 5 → Verify the change on your Student Portal.

⚠️ Schedule changes are only allowed during the official Adding/Dropping Period (first 2 weeks of the semester).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 CLASS SUSPENSION ANNOUNCEMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Always monitor:
📱 PLMun Official Facebook Page: Pamantasan ng Lungsod ng Muntinlupa
🌐 PLMun Website: www.plmun.edu.ph
📌 School Bulletin Boards

📞 For concerns: 02-8248-9161 | 📧 plmuncomm@plmun.edu.ph
Do you have more questions about your schedule? 😊""",
    ],

    "policies": [
        """📜 PLMun GRADING SYSTEM & ACADEMIC POLICIES

Here is a detailed guide based on the PLMun Student Handbook:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 PLMun GRADING SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLMun uses a numerical grading system:

1.00 — Excellent        (97–100%)
1.25 — Superior         (94–96%)
1.50 — Very Good        (91–93%)
1.75 — Good             (88–90%)
2.00 — Meritorious      (85–87%)
2.25 — Very Satisfactory(82–84%)
2.50 — Satisfactory     (79–81%)
2.75 — Fairly Satisfactory (76–78%)
3.00 — Passing          (75%)
5.00 — Failure          (below 75%)
INC  — Incomplete       (requirements not yet completed)
DRP  — Dropped          (officially dropped during dropping period)

✅ Passing Grade: 3.00 (equivalent to 75%)
❌ Failing Grade: 5.00 (below 75%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 ATTENDANCE POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Maximum allowed absences: 20% of the total class hours per subject.
• Exceeding the allowed absences may result in a grade of DROPPED (DRP) or FAILED (5.00).
• Always communicate with your professor for valid absences.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 INC (INCOMPLETE) GRADE — WHAT TO DO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Contact your professor immediately to identify missing requirements.
Step 2 → Complete and submit the missing requirements within the set deadline.
Step 3 → Your professor will submit a grade change to the Registrar's Office.
Step 4 → Check your Student Portal after a few days to see your updated grade.

⚠️ Unresolved INC grades may convert to 5.00 (Failing) after the deadline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 LATIN HONORS — PLMun REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏅 Summa Cum Laude — GWA of 1.00 to 1.25
   (Exempted from tuition fees for one semester if re-enrolling for another degree)
🥈 Magna Cum Laude — GWA of 1.26 to 1.50
   (Exempted from tuition fees for one semester if re-enrolling for another degree)
🥉 Cum Laude        — GWA of 1.51 to 1.75
   (Entitled to 50% tuition fee discount for one semester if re-enrolling)

Requirements: No failing grades, no dropped subjects (verify current requirements with the Registrar).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO APPEAL A GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Talk to your Professor and request a grade breakdown.
Step 2 → If unresolved, write a formal Grade Appeal Letter to the Department Chair.
Step 3 → Attach supporting evidence (quiz papers, project receipts, etc.).
Step 4 → If still unresolved, escalate to the Dean's Office.
Step 5 → The Dean will review and issue a final decision.

⚠️ File your appeal within the appeal period stated in the PLMun Student Handbook.

📖 Full policies are in the PLMun Student Handbook — ask the Registrar's Office for a copy.
📞 02-8248-9161 | 🌐 www.plmun.edu.ph
Do you have a specific policy concern? 😊""",

        """📜 PLMun DROPPING, SHIFTING & LEAVE OF ABSENCE GUIDE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO DROP A SUBJECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Visit the Registrar's Office and get an Adding/Dropping Form.
Step 2 → Fill out the form with the subject(s) you want to drop.
Step 3 → Have it signed by your Professor and Academic Adviser.
Step 4 → Submit to the Registrar's Office BEFORE the dropping deadline.
   (Dropping period: first 2 weeks of the semester)
Step 5 → Keep a copy of your dropping form for your records.

⚠️ Dropping AFTER the deadline = 5.00 (Failing) on your record.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO SHIFT TO ANOTHER COURSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Talk to your Academic Adviser about your plan to shift.
Step 2 → Visit the Registrar's Office and request a Shifting Form.
Step 3 → Get approval from your current Department Chair.
Step 4 → Visit the department of your new course and get acceptance approval.
Step 5 → Submit all signed forms to the Registrar's Office.
Step 6 → Re-enroll under your new course during the next enrollment period.

⚠️ Shifting is subject to availability of slots and approval of both departments.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO APPLY FOR LEAVE OF ABSENCE (LOA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Get a Leave of Absence (LOA) Form from the Registrar's Office.
Step 2 → Fill out the form and clearly state your reason for the LOA.
Step 3 → Have it approved by your Academic Adviser and Dean.
Step 4 → Submit the completed form to the Registrar's Office.
Step 5 → Keep a copy and note your return-to-study deadline.

⚠️ LOA is subject to approval and is limited by PLMun's maximum residency policy.
⚠️ Maximum residency at PLMun: 1.5 times the normal program length (e.g., 6 years for a 4-year course).

📞 02-8248-9161 | 🌐 www.plmun.edu.ph | 📧 universityregistrar@plmun.edu.ph
Do you have more policy questions? 😊""",
    ],

    "documents": [
        """📁 PLMun DOCUMENT REQUEST GUIDE

Here is a step-by-step guide to requesting official documents from the PLMun Registrar's Office:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 DOCUMENTS YOU CAN REQUEST AT PLMun
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Transcript of Records (TOR)
✅ Certificate of Enrollment / Registration (COR)
✅ Certificate of Good Moral Character
✅ Certificate of Graduation
✅ Certified True Copy of Grades
✅ Diploma (original or replacement)
✅ Honorable Dismissal / Transfer Credentials
✅ Form 137 (Secondary School Records)
✅ School Clearance
✅ Certificate of Units Earned
✅ Certificate of Medium of Instruction
✅ Authentication of Documents

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEPS TO REQUEST A DOCUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Go to the PLMun Registrar's Office
   📍 Location: Main Building, PLMun Campus, University Road, NBP Reservation, Brgy. Poblacion, Muntinlupa City
   🕐 Office Hours: Monday – Friday, 8:00 AM – 5:00 PM
   📧 Email: universityregistrar@plmun.edu.ph

Step 2 → Fill out a Document Request Form. Specify:
   • Type of document needed
   • Number of copies
   • Purpose (employment, board exam, scholarship, etc.)

Step 3 → Present your valid Student ID or government-issued ID.

Step 4 → Go to the Treasury / Cashier's Office to pay the document fee.
   • Fees vary per document type — ask the Registrar's Office for the current fee schedule.

Step 5 → Return to the Registrar's Office and submit your Official Receipt.

Step 6 → Wait for processing:
   • Regular: 3 to 5 working days
   • Rush processing: may be available at additional cost (inquire at Registrar's Office)

Step 7 → Return on the scheduled release date to claim your document.
   • Bring your Official Receipt and valid ID.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 PROXY CLAIMING (If someone will claim for you)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your representative must bring:
✅ Authorization Letter (signed by you)
✅ Photocopy of your valid ID
✅ Their own valid ID (original)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 FOR BOARD EXAM APPLICANTS (NLE, LET, CPA, Criminology Board, etc.)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Typically required from PLMun Registrar:
✅ Transcript of Records (TOR) — certified true copy
✅ Certificate of Graduation
✅ Diploma (original)
✅ Certificate of Good Moral Character

⚠️ Request at least 2–3 weeks before your board exam application deadline!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 LOST DIPLOMA — REPLACEMENT PROCEDURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Prepare a notarized Affidavit of Loss.
Step 2 → Submit it to the Registrar's Office with your valid ID.
Step 3 → Pay the Diploma Replacement Fee at the Treasury / Cashier's Office.
Step 4 → Wait for the processing and claim on the given schedule.

📞 02-8248-9161 loc. 146 | 📧 universityregistrar@plmun.edu.ph | 🌐 www.plmun.edu.ph
Do you need help with a specific document request? 😊""",
    ],

    "services": [
        """🎓 PLMun SCHOLARSHIP GUIDE — Iskolar ng Bayan & More

Here is a complete guide to scholarships and financial aid at PLMun:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 AVAILABLE SCHOLARSHIPS AT PLMun
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏛️ ISKOLAR NG BAYAN SCHOLARSHIP PROGRAM
   • Established by City Ordinance of Muntinlupa (Ordinance Nos. 98-08, 08-042, 13-010)
   • Provides FREE college education and incentives for qualified Muntinlupa residents.
   • Benefits: Free tuition, allowances, and other incentives depending on the ordinance.

🎓 UNIFAST / RA 10931 (Universal Access to Quality Tertiary Education Act)
   • PLMun students enjoy FREE TUITION under this national law.
   • Miscellaneous fees may still apply — inquire at the Cashier's Office.

🏫 PLMun ACADEMIC EXCELLENCE SCHOLARSHIP
   • Merit-based — for students with outstanding academic performance.
   • Magna Cum Laude graduates: Exempt from tuition for 1 semester (if re-enrolling).
   • Cum Laude graduates: 50% tuition discount for 1 semester (if re-enrolling).

🤝 OUT-SOURCED SCHOLARSHIPS (inquire at the Student Affairs Office / OSA):
   ✅ Iskolar ng Bayan Councilor's Scholarship
   ✅ How Good Foundation, Inc.
   ✅ P.D. 577 — Veterans' Beneficiaries
   ✅ LCCK Foundation
   ✅ Charity First Foundation
   ✅ CHED / PESFA / UNIFAST Grants

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEPS TO APPLY FOR A SCHOLARSHIP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Visit the PLMun Student Affairs Office (OSA) and ask for the list of available scholarships.
Step 2 → Get the Scholarship Application Form from the OSA.
Step 3 → Prepare the common requirements:
   ✅ Duly filled-out Application Form
   ✅ Certificate of Registration (COR) or Enrollment Form
   ✅ Certified True Copy of Grades / TOR
   ✅ General Point Average (GPA) Certificate
   ✅ Certificate of Good Moral Character
   ✅ Barangay Certificate of Indigency (for need-based scholarships)
   ✅ Proof of Family Income (ITR or Certificate of No Income)
   ✅ Certificate of Matriculation (COM) — submitted to the Scholarship Coordinator within 1 week after enrollment
   ✅ Barangay Clearance
   ✅ 2x2 ID photos (2 copies)

Step 4 → Submit complete requirements to the Scholarship Office / OSA before the deadline.
Step 5 → Wait for the evaluation and approval.
Step 6 → Once approved, maintain the required GWA to keep your scholarship.

⚠️ Scholarship slots are limited — apply early! Deadlines are strictly observed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 PLMun STUDENT SERVICES DIRECTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏥 Medical / Dental Clinic — Free health services for enrolled students (bring Student ID)
📚 eLibrary — Online library resources at: https://plmun-library.follettdestiny.com
💬 Guidance & Counseling Office — Free, confidential counseling (walk-in or appointment)
🪪 Student Affairs Office (OSA) — Scholarships, org, student concerns
💻 ICT Office — Student Portal and tech concerns | 📧 ict@plmun.edu.ph
📞 Support: support@plmun.edu.ph

📞 PLMun Main: 02-8248-9161 loc. 146
📧 Email: plmuncomm@plmun.edu.ph
🌐 Website: www.plmun.edu.ph
📍 Address: University Road, NBP Reservation, Brgy. Poblacion, Muntinlupa City, 1776

Do you want more details about a specific service or scholarship? 😊""",

        """🪪 PLMun STUDENT ID & PORTAL GUIDE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO GET YOUR STUDENT ID
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → After enrollment, go to the PLMun Student Affairs Office (OSA).
Step 2 → Bring:
   ✅ Certificate of Registration (COR) or Enrollment Form
   ✅ 1x1 or 2x2 ID photo (white background, 2 copies)
   ✅ Official Receipt of ID fee payment
Step 3 → Fill out the ID Application Form.
Step 4 → Wait for your ID to be processed and released.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO REPLACE A LOST ID
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Report the loss to the Security Office or Student Affairs Office.
Step 2 → Prepare a notarized Affidavit of Loss.
Step 3 → Pay the ID Replacement Fee at the Treasury / Cashier's Office.
Step 4 → Submit the Affidavit of Loss and Official Receipt to the Student Affairs Office.
Step 5 → Wait for your replacement ID.

⚠️ Wearing your PLMun Student ID inside the campus is required by school policy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO ACCESS THE PLMun STUDENT PORTAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔗 URL: https://plmun.edu.ph/student-portal/system/main/index.php
• Log in using your PLMun Student ID Number and password.
• Use the portal to check grades, view schedule, and monitor enrollment status.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 FORGOT PASSWORD / CANNOT LOGIN?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Visit the PLMun ICT Office or Registrar's Office in person.
Step 2 → Provide your Student ID Number and valid ID for verification.
Step 3 → Request a password reset.
Step 4 → The ICT staff will reset your account and give you a temporary password.
Step 5 → Log in and change your password immediately.

📧 ICT Support: ict@plmun.edu.ph | support@plmun.edu.ph
📞 02-8248-9161 | 🌐 www.plmun.edu.ph

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 PLMun eLIBRARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔗 Access: https://plmun-library.follettdestiny.com
• Available to all enrolled PLMun students.
• Physical library is open during school hours.
• Bring your Library Card / Student ID to borrow books.

Do you need help with other student services? 😊""",
    ],

    "unknown": [
        """🤔 I'm sorry, I wasn't able to fully understand your question.

Don't worry! Here's what you can do:

📌 Try rephrasing your question. Examples:
   • "How do I apply at PLMun?" → Enrollment & PCAT info
   • "What courses are offered?" → PLMun academic programs
   • "How do I get my TOR?" → Document request guide
   • "Is there a scholarship at PLMun?" → Iskolar ng Bayan info
   • "What is the passing grade?" → Grading system info

📌 Topics I can best help you with:
   📋 Enrollment & Admission (PCAT, requirements, courses)
   📅 Class Schedule & Academic Calendar
   📜 School Policies (grading, attendance, conduct)
   📁 Documents (TOR, certificates, diploma)
   🎓 Student Services (scholarships, guidance, portal, ID)

📌 For urgent concerns, contact PLMun directly:
   📞 Telephone: 02-8248-9161 loc. 146
   📧 Email: plmuncomm@plmun.edu.ph
   🌐 Website: www.plmun.edu.ph
   📍 Address: University Road, NBP Reservation, Brgy. Poblacion, Muntinlupa City, 1776

Feel free to try again — I'm here to help! 😊""",
    ]
}

def get_response(intent):
    return random.choice(RESPONSES.get(intent, RESPONSES["unknown"]))

def train_model(texts, labels):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ('nb', MultinomialNB(alpha=0.3))
    ])
    pipeline.fit([preprocess(t) for t in texts], labels)
    return pipeline

def evaluate_model(texts, labels, pipeline):
    processed = [preprocess(t) for t in texts]
    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.2, random_state=42, stratify=labels)
    ep = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=1)), ('nb', MultinomialNB(alpha=0.3))])
    ep.fit(X_train, y_train)
    y_pred = ep.predict(X_test)
    intents = sorted(set(labels))
    cv = cross_val_score(ep, processed, labels, cv=5, scoring='accuracy')
    return {
        "accuracy": round(accuracy_score(y_test, y_pred)*100, 2),
        "precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0)*100, 2),
        "recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0)*100, 2),
        "f1_score": round(f1_score(y_test, y_pred, average='weighted', zero_division=0)*100, 2),
        "cross_val_mean": round(cv.mean()*100, 2),
        "cross_val_std": round(cv.std()*100, 2),
        "cv_scores": [round(s*100, 2) for s in cv],
        "train_size": len(X_train), "test_size": len(X_test), "total_samples": len(labels),
        "intents": intents,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=intents).tolist(),
        "report": classification_report(y_test, y_pred, target_names=intents, output_dict=True)
    }

print("Loading dataset...", end=" ")
texts, labels = load_dataset(DATASET_FILE)
print(f"Done. ({len(texts)} samples)")
print("Training model...", end=" ")
pipeline = train_model(texts, labels)
print("Done.")
eval_results = evaluate_model(texts, labels, pipeline)
print(f"Accuracy: {eval_results['accuracy']}% | F1: {eval_results['f1_score']}% | CV: {eval_results['cross_val_mean']}%")
print("PLMun Chatbot Ready! 💚")

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
    method, confidence = "Rule-Based", 100.0
    if intent is None:
        probs = pipeline.predict_proba([cleaned])[0]
        idx = probs.argmax()
        intent = pipeline.classes_[idx]
        confidence = round(float(probs[idx])*100, 1)
        method = "Naive Bayes"
    return jsonify({
        "response": get_response(intent), "intent": intent,
        "method": method, "confidence": confidence,
        "response_time_ms": round((time.time()-start)*1000, 2)
    })

@app.route("/eval")
def get_eval():
    return jsonify(eval_results)

@app.route("/stats")
def get_stats():
    return jsonify({
        "total_samples": len(texts), "total_intents": len(set(labels)),
        "intents": sorted(set(labels)),
        "algorithm": "Hybrid: Rule-Based + TF-IDF + Multinomial Naive Bayes",
        "accuracy": eval_results["accuracy"], "f1_score": eval_results["f1_score"]
    })

def chatbot_respond(user_input, vectorizer, model):
    cleaned = preprocess(user_input)

    intent = rule_based_classify(cleaned)
    method = "Rule-Based"
    confidence = 100.0

    if intent is None:
        probs = model.predict_proba([cleaned])[0]
        idx = probs.argmax()
        intent = model.classes_[idx]
        confidence = round(float(probs[idx]) * 100, 1)
        method = "Naive Bayes"

    response = get_response(intent)

    return response, intent, method

if __name__ == "__main__":
    app.run(debug=True)
