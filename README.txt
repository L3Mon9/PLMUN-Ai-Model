# AI-Powered Institutional Support Chatbot
**Thesis Project — Hybrid NLP Intent Classifier**

> Development of an AI-Powered Chatbot Using Natural Language Processing and  
> Intent Classification for Institutional Inquiry and Student Support Services

---

## Project Structure

```
chatbot_project/
├── chatbot.py        ← Main program (run this)
├── dataset.csv       ← 100 labeled training samples
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

---

## Requirements

- Python 3.8 or higher
- pip (Python package manager)

---

## Installation

**Step 1 — Open a terminal and navigate to the project folder:**
```bash
cd chatbot_project
```

**Step 2 — Install the required library:**
```bash
pip install -r requirements.txt
```

---

## Running the Chatbot

```bash
python chatbot.py
```

You will see:
```
  AI Chatbot Ready!
```

Then type any question and press **Enter**.

---

## Sample Interactions

```
  You: how do i enroll as a transferee
  Bot [Rule-Based → enrollment]:
  For enrollment inquiries, please visit the Registrar's Office...

  You: when does the semester start
  Bot [Rule-Based → schedule]:
  Your class schedule is available through the Student Portal...

  You: what is the grading system
  Bot [Rule-Based → policies]:
  The school's academic policies are outlined in the Student Handbook...
```

---

## Special Commands

| Command | Description                        |
|---------|------------------------------------|
| `eval`  | Run model accuracy evaluation      |
| `exit`  | Quit the chatbot                   |

---

## How the Model Works

This chatbot uses a **Hybrid Classification** approach:

### Layer 1 — Rule-Based Keyword Matching
- Scans the user's input for predefined keywords.
- If a keyword match is found, the intent is immediately identified.
- Fast, transparent, and reliable for common queries.

### Layer 2 — Multinomial Naive Bayes (ML Fallback)
- Used when keyword matching returns no result.
- Text is converted into numerical features using **Bag-of-Words** (CountVectorizer with bigrams).
- The Naive Bayes model applies Bayes' theorem to compute the probability of each intent and selects the most likely one.

### Why Naive Bayes Has No Epochs
Naive Bayes is a **probabilistic, non-iterative** algorithm. It calculates class probabilities and word likelihoods directly from the training data in a single pass using counting and statistics — no gradient descent, no weight updates, and therefore no epochs.

---

## Supported Intents

| Intent       | Description                                  |
|--------------|----------------------------------------------|
| `enrollment` | Admission, registration, tuition, units      |
| `schedule`   | Class times, semester dates, academic calendar |
| `policies`   | Grades, attendance, conduct, appeals         |
| `documents`  | TOR, certificates, registrar requests        |
| `services`   | Scholarships, guidance, portals, clinics     |

---

## Notes

- Fully **offline** — no internet or API required.
- All training data is in `dataset.csv` — you can add more rows anytime.
- Re-running `python chatbot.py` retrains the model automatically from the CSV.
