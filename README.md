# MLPilot 🚀

An **Agentic MLOps Platform** designed to automate the boring 80% of data science (Data Profiling, Feature Engineering, and Baseline Modeling) while leaving the human in total control. MLPilot marries the creative reasoning of LLMs with the deterministic mathematical execution of Scikit-Learn and XGBoost.

---

## 💡 What is the Business Value? (Why build this?)
Traditional AutoML tools (like DataRobot or H2O.ai) are massively expensive "Black Boxes." You upload data, they test 10,000 combinations using brute-force compute, and spit out a model. If a bank denies a loan using that model, they can't legally explain *why*. 

**MLPilot is different. It is a "White-box Agentic Workflow."**
1. **Intelligent Ideation, not Brute Force:** Instead of testing 10,000 random math formulas, we pass the dataset *metadata* to an LLM. The LLM reads the column names (e.g. `TicketPrice` and `Age`) and intelligently suggests exactly 3 highly-logical feature ideas (e.g., `Wealth_Index`). 
2. **Transparent Execution:** The LLM does *no math*. It simply writes a JSON blueprint. The backend Python engine parses the blueprint, applies the Scikit-Learn transformations, and trains an XGBoost model.
3. **Auditability:** The human data scientist sees the LLM's logic, approves the experiment, and has the exact Python logic on hand for legal compliance.

---

## 🧠 Core ML Concepts (For Beginners / Interview Prep)

If you haven't touched ML in a while, here is your cheat sheet for this codebase:

* **Target Leakage:** The deadliest bug in ML. It happens when your training data includes a column that allows the model to "cheat" because it wouldn't exist in the real world. (e.g., trying to predict who survived the Titanic, but accidentally leaving a column called `Lifeboat_Number` in the data. The model just learns "If they have a lifeboat, they survive!"). **MLPilot automatically scans for this and flags it.**
* **Baseline Model:** The absolute simplest model you can train (no feature engineering) to establish a "floor" score that we must try to beat.
* **Feature Engineering:** Creating new mathematical columns out of old ones (e.g., `SibSp` + `Parch` = `Family_Size`).
* **F1 Score:** Our primary metric. Accuracy is misleading (e.g., if 99% of transactions are legit, a model that guesses "legit" every time is 99% accurate but catches 0 fraud). F1 balances Precision (when I say fraud, is it fraud?) and Recall (did I catch all the fraud?).
* **XGBoost:** The undisputed king of tabular (spreadsheet) data. It builds a sequence of decision trees where each tree tries to fix the mistakes of the previous one. We use this as our core engine.

---

## 🛠️ Architecture: The "Brain vs Muscle" Pattern

**1. The Brain (AI Gateway)** 🧠
* Stored in `backend/ai/`. We use a smart `TaskRouter` that maps different tasks to the best LLMs.
* *Current Priority Chain:* Groq (`qwen`/`gpt-oss`) -> Gemini (`flash`) -> NVIDIA NIM (`nemotron`).
* It strictly outputs JSON.

**2. The Muscle (ML Engine)** 💪
* Stored in `backend/ml/`. 
* Uses `asyncio.to_thread` to ensure that heavy XGBoost training never blocks the FastAPI server loop.

---

## 🚀 How to Run (Quick Start)

We have a unified start script to launch both the backend and frontend simultaneously:

```bash
python start.py
```

* **Frontend:** http://localhost:5173
* **Backend API:** http://localhost:8000
* **API Keys:** Add them to `backend/.env`. If an API key rate-limits, the system safely falls back to the `StubProvider` (offline mode).

---

## 📈 Future Milestones (Where this goes next)
<!-- Add new milestones here -->
