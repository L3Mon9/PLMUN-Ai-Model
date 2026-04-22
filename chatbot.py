# =============================================================================
# FLASK WEB APP — AI Institutional Support Chatbot
# PLMun — Pamantasan ng Lungsod ng Muntinlupa
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
        "miscellaneous fee", "form 138", "placement test", "regular student",
        "irregular student", "cross enroll", "quota"
    ],
    "schedule": [
        "schedule", "class", "classes", "semester", "academic calendar",
        "timetable", "calendar", "holiday", "semestral break",
        "midterm", "final exam", "first day of class", "make up class",
        "class suspended", "last day of class", "school day",
        "kelan ang", "kailan magsisimula", "may pasok", "anong oras",
        "dropping period", "summer class", "exam week", "recognition day",
        "graduation", "school year", "christmas vacation", "fiesta",
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
        "shuttle", "parking", "school location", "campus",
        "mental health", "facebook", "website", "email",
        "contact number", "office hours", "student council",
        "join club", "student organization", "work study", "part time",
        "CHED scholarship", "indigency", "tuition waiver", "financial assistance",
        "reset password", "cannot login", "student portal problem"
    ]
}

def rule_based_classify(text):
    for intent, keywords in KEYWORD_MAP.items():
        for keyword in keywords:
            if keyword in text:
                return intent
    return None

# =============================================================================
# RESPONSES — Smart, Long, Step-by-Step, Easy to Understand
# =============================================================================

RESPONSES = {

    "greeting": [
        """👋 Hello, Iskolar! Welcome to the PLMun Student Support Chatbot!

I am your official AI-powered assistant for Pamantasan ng Lungsod ng Muntinlupa (PLMun). I am available 24/7 to help you with your school-related concerns.

📌 Here are the topics I can help you with:
• 📋 Enrollment — requirements, procedures, fees, and schedules
• 📅 Class Schedule — academic calendar, exam dates, and semestral breaks
• 📜 School Policies — grading system, attendance rules, and academic conduct
• 📁 Documents — TOR, certificates, and other official records
• 🎓 Student Services — scholarships, guidance, clinic, library, and more

💡 Tip: Just type your question naturally, and I will do my best to assist you!

How can I help you today, Iskolar? 😊""",

        """🎓 Mabuhay, Iskolar ng PLMun! I'm your Student Support Chatbot.

Whether you have questions about enrollment, your class schedule, school policies, document requests, or student services — I'm here to guide you every step of the way!

Feel free to type your concern or question below. I'll give you clear, step-by-step guidance so you know exactly what to do. 😊

What would you like to know today?""",
    ],

    "farewell": [
        """👋 Thank you for using the PLMun Student Support Chatbot!

It was a pleasure assisting you today. Here are a few reminders before you go:

✅ Always check the PLMun official website for the latest announcements.
✅ Monitor your Student Portal regularly for grades and enrollment updates.
✅ Don't hesitate to visit the Registrar's Office or Student Affairs Office for concerns that need personal attention.

Good luck with your studies, Iskolar! You've got this! 💪
Feel free to come back anytime you have questions. Goodbye! 😊""",

        """See you, Iskolar! 👋

Remember — the PLMun community is always here to support you. If you have more questions in the future, don't hesitate to come back and ask.

Take care, stay safe, and keep pushing forward in your academic journey! 🎓✨""",
    ],

    "thanks": [
        """😊 You're very welcome, Iskolar!

I'm always here to help you navigate your academic journey at PLMun. Here are a few things you can do next:

✅ If your concern has been resolved — great! Go ahead and take the next steps.
✅ If you need more information — feel free to ask a follow-up question.
✅ If you need to visit an office — make sure to bring your Student ID and required documents.

Is there anything else you'd like to know? I'm happy to help! 🎓""",

        """Glad I could help! 😊

Remember, you can always come back here if you have more questions about enrollment, schedules, policies, documents, or student services.

Good luck, Iskolar! You're doing great! 💪""",
    ],

    "enrollment": [
        """📋 ENROLLMENT GUIDE — PLMun

Here is a complete step-by-step guide on how to enroll at Pamantasan ng Lungsod ng Muntinlupa (PLMun):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 1 — Check the Enrollment Schedule
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Visit the PLMun official website or Facebook page.
• Check the posted Academic Calendar for the enrollment period.
• Take note of the exact dates for your year level or college.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 2 — Prepare Your Documents
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For NEW STUDENTS (Freshmen):
   ✅ Form 138 or Form 137 (Report Card)
   ✅ PSA Birth Certificate (original + photocopy)
   ✅ Certificate of Good Moral Character
   ✅ Medical Certificate from a licensed physician
   ✅ 2x2 ID photos (at least 2 copies)

For TRANSFEREES:
   ✅ Honorable Dismissal from previous school
   ✅ Transcript of Records (certified true copy)
   ✅ PSA Birth Certificate
   ✅ Certificate of Good Moral Character
   ✅ Medical Certificate

For OLD / CONTINUING STUDENTS:
   ✅ Valid Student ID
   ✅ Previous semester's grades or registration form

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 3 — Visit the Registrar's Office
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Go to the Registrar's Office during the enrollment period.
• Submit your documents for verification.
• Get your enrollment assessment form.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 4 — Pay Your Tuition and Fees
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Proceed to the Cashier's Office with your assessment form.
• Pay in full or inquire about installment payment options.
• Keep your Official Receipt — you will need this throughout the semester.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEP 5 — Confirm Your Enrollment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Return to the Registrar's Office with your Official Receipt.
• Get your Certificate of Registration (COR) or enrollment confirmation.
• Check your Student Portal to verify that your subjects are reflected.

⚠️ IMPORTANT REMINDERS:
• Enroll within the official period to avoid late enrollment penalties.
• Incomplete documents may delay or prevent your enrollment.
• Online enrollment may be available — check the PLMun Student Portal.

Do you have more questions about enrollment? Feel free to ask! 😊""",

        """📋 TUITION AND FEES GUIDE — PLMun

Here is what you need to know about tuition and payment at PLMun:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 PAYMENT OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLMun offers flexible payment schemes for students:

✅ Full Payment — Pay the total amount at once at the Cashier's Office.
✅ Installment Payment — Pay in staggered amounts. Ask the Cashier's Office about the available installment schedule.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEPS TO PAY YOUR TUITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Get your Assessment Form from the Registrar's Office after enrollment assessment.
Step 2 → Proceed to the Cashier's Office with your Assessment Form.
Step 3 → Choose your payment option (full or installment).
Step 4 → Pay and receive your Official Receipt.
Step 5 → Present your Official Receipt to the Registrar to complete your enrollment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎓 SCHOLARSHIP OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If you're having difficulty paying tuition, you may apply for:
• Government scholarships (CHED, PESFA, UNIFAST)
• PLMun institutional scholarships
• Local Government Unit (LGU) grants

Visit the Student Affairs Office for scholarship application details.

⚠️ Always keep your Official Receipt — it is proof of your payment and needed for enrollment completion.

Would you like to know more about scholarships or enrollment requirements? 😊""",
    ],

    "schedule": [
        """📅 CLASS SCHEDULE GUIDE — PLMun

Here is how you can find and manage your class schedule at PLMun:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO CHECK YOUR CLASS SCHEDULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Log in to the PLMun Student Portal using your student credentials.
Step 2 → Navigate to the Schedule or Enrollment section.
Step 3 → View your list of enrolled subjects and their schedules.
Step 4 → Take note of your room assignments and class times.

If you cannot access the Student Portal:
→ Visit the Registrar's Office and request a printed copy of your schedule.
→ Bring your Student ID for verification.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 ACADEMIC CALENDAR OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLMun follows a semester-based academic year:

📘 First Semester — typically starts in August
📗 Second Semester — typically starts in January
☀️ Summer Term — available for selected subjects (usually March to May)

Key dates included in the Academic Calendar:
✅ First Day of Classes
✅ Midterm Examination Week
✅ Final Examination Week
✅ Semestral Break
✅ Enrollment Period
✅ Dropping / Adding Period
✅ School Holidays and Special Events

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO REQUEST A SCHEDULE CHANGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Talk to your Academic Adviser about your concern.
Step 2 → Fill out a Schedule Change Request form from the Registrar's Office.
Step 3 → Get approval from your Department Chair or Dean.
Step 4 → Submit the approved form to the Registrar's Office.
Step 5 → Check your updated schedule on the Student Portal.

⚠️ REMINDER: Schedule changes are only allowed during the official adding/dropping period.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 CLASS SUSPENSION ANNOUNCEMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For class suspension updates, always monitor:
📱 PLMun Official Facebook Page
🌐 PLMun Official Website
🗓️ Student Portal Announcements
📌 School Bulletin Boards

Do you have more questions about your schedule or the academic calendar? 😊""",
    ],

    "policies": [
        """📜 GRADING SYSTEM AND ATTENDANCE POLICY — PLMun

Here is a detailed guide to understanding PLMun's academic policies:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 GRADING SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLMun uses a numerical grading system:

Grade 1.00 — Excellent (99–100%)
Grade 1.25 — Superior (96–98%)
Grade 1.50 — Very Good (93–95%)
Grade 1.75 — Good (90–92%)
Grade 2.00 — Meritorious (87–89%)
Grade 2.25 — Very Satisfactory (84–86%)
Grade 2.50 — Satisfactory (81–83%)
Grade 2.75 — Fairly Satisfactory (78–80%)
Grade 3.00 — Passing (75–77%)
Grade 5.00 — Failure (below 75%)
INC — Incomplete (requirements not complete)

✅ Passing Grade: 3.00 or higher (equivalent to 75%)
❌ Failing Grade: 5.00 (below 75%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 ATTENDANCE POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Maximum allowed absences: 20% of total class hours per subject.
• Example: If a subject meets 3 hours per week × 18 weeks = 54 hours total → Maximum absences = 10.8 hours (approximately 3–4 sessions).
• Exceeding the allowed absences may result in a grade of DROPPED (D) or FAILED (5.00).

⚠️ Always communicate with your professor if you need to be absent for valid reasons.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 WHAT TO DO IF YOU RECEIVE AN INC GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Contact your professor immediately to know what requirements are missing.
Step 2 → Complete and submit the missing requirements within the set deadline.
Step 3 → Your professor will submit your final grade to the Registrar's Office.
Step 4 → Check your Student Portal after a few days to see your updated grade.

⚠️ If the INC is not resolved within the allowed period, it may automatically become a failing grade (5.00).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 LATIN HONORS REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏅 Summa Cum Laude — GWA of 1.00 to 1.25 (no failing grades, no dropped subjects)
🥈 Magna Cum Laude — GWA of 1.26 to 1.50
🥉 Cum Laude        — GWA of 1.51 to 1.75

Please verify current requirements with the Registrar's Office as standards may be updated.

Do you have a specific policy concern you'd like to know more about? 😊""",

        """📜 GRADE APPEAL AND DROPPING POLICY — PLMun

Here is what you need to know about grade appeals and dropping subjects:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO APPEAL A GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If you believe your grade is incorrect or unfair, here are the steps:

Step 1 → Talk to your Professor first. Ask for a breakdown of your grades.
Step 2 → If unresolved, write a formal Grade Appeal Letter addressed to the Department Chair.
Step 3 → Submit the letter to the Department Chair's Office along with supporting evidence (e.g., graded outputs, quiz papers).
Step 4 → If still unresolved, escalate to the Dean's Office.
Step 5 → The Dean will review and make a final decision.

⚠️ File your appeal within the appeal period stated in the Student Handbook. Late appeals may not be accepted.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO DROP A SUBJECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Visit the Registrar's Office and get a Subject Dropping Form.
Step 2 → Fill out the form with the subject(s) you want to drop.
Step 3 → Get the signature of your Professor and Academic Adviser.
Step 4 → Submit the completed form to the Registrar's Office before the dropping deadline.
Step 5 → Keep a copy of your dropping form for your records.

⚠️ Dropping after the deadline may result in a failing grade (5.00) on your record.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO APPLY FOR LEAVE OF ABSENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Secure a Leave of Absence (LOA) Form from the Registrar's Office.
Step 2 → Fill out the form and state your reason clearly.
Step 3 → Get approval from your Academic Adviser and Dean.
Step 4 → Submit the approved form to the Registrar's Office.
Step 5 → Keep a copy for your records and monitor your re-enrollment schedule.

⚠️ LOA is subject to approval and is typically limited to a certain number of semesters.

Would you like more information about school policies? 😊""",
    ],

    "documents": [
        """📁 DOCUMENT REQUEST GUIDE — PLMun Registrar's Office

Here is a complete step-by-step guide on how to request official documents at PLMun:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 AVAILABLE DOCUMENTS YOU CAN REQUEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Transcript of Records (TOR)
✅ Certificate of Enrollment / Registration
✅ Certificate of Good Moral Character
✅ Certificate of Graduation
✅ Certified True Copy of Grades
✅ Diploma (original release or replacement)
✅ Honorable Dismissal / Transfer Credentials
✅ Form 137 (for senior high or elementary records)
✅ School Clearance
✅ Certificate of Units Earned
✅ Authentication of Documents

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEPS TO REQUEST A DOCUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Go to the PLMun Registrar's Office during office hours (Monday–Friday, 8:00 AM – 5:00 PM).

Step 2 → Fill out a Document Request Form. Specify:
   • Type of document you need
   • Number of copies
   • Purpose (e.g., for employment, board exam, scholarship)

Step 3 → Present your valid Student ID or any government-issued ID for verification.

Step 4 → Proceed to the Cashier's Office to pay the document fee.
   • Document fees vary depending on the type — ask the Registrar for the current fee schedule.

Step 5 → Return to the Registrar's Office and submit your Official Receipt.

Step 6 → Wait for the processing period:
   • Regular processing: 3 to 5 working days
   • Rush processing: may be available for an additional fee (inquire at the Registrar)

Step 7 → Return to the Registrar's Office on the scheduled release date to claim your document.
   • Bring your Official Receipt and valid ID.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 IF SOMEONE ELSE WILL CLAIM FOR YOU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your representative must bring:
✅ Authorization Letter (signed by you)
✅ Photocopy of your valid ID
✅ Their own valid ID (original)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 FOR BOARD EXAM APPLICANTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Typically required documents:
✅ Transcript of Records (TOR)
✅ Certificate of Graduation
✅ Diploma
✅ Good Moral Character Certificate

⚠️ Request your documents at least 2 to 3 weeks before your board exam application deadline to account for processing time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 FOR LOST DIPLOMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Prepare an Affidavit of Loss (notarized).
Step 2 → Submit it to the Registrar's Office along with your valid ID.
Step 3 → Pay the Diploma Replacement Fee at the Cashier's Office.
Step 4 → Wait for the processing period and claim your replacement on the given schedule.

Do you have a specific document you need help requesting? 😊""",
    ],

    "services": [
        """🎓 SCHOLARSHIP APPLICATION GUIDE — PLMun

Here is a complete guide on how to apply for scholarships and financial assistance at PLMun:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 AVAILABLE SCHOLARSHIPS AT PLMun
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏛️ Government Scholarships:
   • CHED-UNIFAST Scholarship (Free Higher Education)
   • PESFA (Private Education Student Financial Assistance)
   • GSISEA / PAGIBIG Scholarships (for dependents of members)

🏫 Institutional Scholarships:
   • PLMun Scholarship (merit-based or need-based)
   • Academic Excellence Awards

🏙️ Local Government Scholarships:
   • Muntinlupa City Government Scholarship
   • Barangay-level scholarships

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 STEPS TO APPLY FOR A SCHOLARSHIP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Visit the Student Affairs Office or Scholarship Office at PLMun.
Step 2 → Ask for the list of available scholarships and their requirements.
Step 3 → Prepare the required documents. Common requirements include:
   ✅ Application Form (get from the Scholarship Office)
   ✅ Certificate of Registration (COR) or Enrollment
   ✅ Copy of Grades / TOR
   ✅ Certificate of Good Moral Character
   ✅ Barangay Certificate of Indigency (for need-based)
   ✅ Proof of Family Income (ITR or Certificate of No Income)
   ✅ 2x2 ID photo (2 copies)
   ✅ Photocopy of valid ID

Step 4 → Submit your complete requirements to the Scholarship Office within the deadline.
Step 5 → Wait for the evaluation and results announcement.
Step 6 → If approved, comply with the scholarship conditions (e.g., maintaining a minimum GWA).

⚠️ Scholarship deadlines are strictly followed. Always apply early to avoid missing out!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 WHERE TO GO FOR OTHER SERVICES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏥 Medical / Dental Clinic → For free health services (bring Student ID)
📚 Library → For books, references, and e-resources (bring Library Card)
💬 Guidance Office → For counseling and personal concerns (confidential)
🪪 Student Affairs Office → For ID, organizations, and student concerns
💻 ICT Office → For Student Portal issues and technical concerns
💰 Cashier's Office → For tuition payments and official receipts

Do you want more details about a specific service or scholarship? 😊""",

        """🪪 STUDENT ID GUIDE — PLMun

Here is everything you need to know about getting or replacing your PLMun Student ID:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO GET A NEW STUDENT ID (For New Students)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → After enrollment, go to the Student Affairs Office.
Step 2 → Bring the following:
   ✅ Certificate of Registration (COR) or Enrollment Form
   ✅ 1x1 or 2x2 ID photo (white background)
   ✅ Official Receipt of ID fee payment from the Cashier's Office

Step 3 → Fill out the ID Application Form.
Step 4 → Wait for your ID to be processed and released.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO REPLACE A LOST STUDENT ID
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Prepare an Affidavit of Loss (can be notarized at a notary public).
Step 2 → Report the loss to the Student Affairs Office or Security Office.
Step 3 → Pay the ID Replacement Fee at the Cashier's Office.
Step 4 → Submit the Affidavit of Loss and Official Receipt to the Student Affairs Office.
Step 5 → Wait for the replacement ID to be processed.

⚠️ Always wear your Student ID inside the campus — it is required by PLMun policy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 HOW TO RESET YOUR STUDENT PORTAL PASSWORD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 → Go to the PLMun ICT Office or Registrar's Office.
Step 2 → Provide your Student ID Number and valid ID for verification.
Step 3 → Request a password reset.
Step 4 → The staff will reset your account and provide a temporary password.
Step 5 → Log in and change your password immediately.

💡 You can also check the PLMun website or Student Portal login page for a "Forgot Password" option.

Do you need help with other student services? 😊""",
    ],

    "unknown": [
        """🤔 I'm sorry, I wasn't able to fully understand your question.

Don't worry! Here's what you can do:

📌 Try rephrasing your question. For example:
   • Instead of "How?" → Try "How do I enroll at PLMun?"
   • Instead of "Documents" → Try "How do I request my Transcript of Records?"

📌 Here are the topics I can best assist you with:
   📋 Enrollment — requirements, procedures, fees
   📅 Schedule — class schedule, academic calendar, exam dates
   📜 Policies — grading, attendance, conduct, appeals
   📁 Documents — TOR, certificates, diplomas
   🎓 Services — scholarships, guidance, library, clinic, ID

📌 If your concern is urgent or requires personal attention, please visit:
   • Registrar's Office — for enrollment and documents
   • Student Affairs Office — for scholarships and student concerns
   • Guidance Office — for personal and academic counseling

Feel free to try again — I'm here to help! 😊""",
    ]
}

def get_response(intent):
    return random.choice(RESPONSES.get(intent, RESPONSES["unknown"]))

def train_model(texts, labels):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, analyzer='word')),
        ('nb', MultinomialNB(alpha=0.3))
    ])
    processed = [preprocess(t) for t in texts]
    pipeline.fit(processed, labels)
    return pipeline

def evaluate_model(texts, labels, pipeline):
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
    cv_scores = cross_val_score(eval_pipeline, processed, labels, cv=5, scoring='accuracy')
    return {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        "cross_val_mean": round(cv_scores.mean() * 100, 2),
        "cross_val_std": round(cv_scores.std() * 100, 2),
        "cv_scores": [round(s * 100, 2) for s in cv_scores],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_samples": len(labels),
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
print(f"Accuracy : {eval_results['accuracy']}%  |  F1: {eval_results['f1_score']}%  |  CV: {eval_results['cross_val_mean']}%")
print("PLMun Chatbot Ready!")

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
        idx = probs.argmax()
        intent = pipeline.classes_[idx]
        confidence = round(float(probs[idx]) * 100, 1)
        method = "Naive Bayes"
    response_time = round((time.time() - start) * 1000, 2)
    return jsonify({
        "response": get_response(intent),
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

def chatbot_respond(user_input, vectorizer, model):
    cleaned = preprocess(user_input)

    # Rule-based muna
    intent = rule_based_classify(cleaned)
    method = "Rule-Based"
    confidence = 100.0

    # If walang match → ML
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