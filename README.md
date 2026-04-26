# Lab Assignment 5: LSTM-Based AI Agent for Sequence Prediction

## Group Members
- **Mohit Patil** (202301040272)
- **Parimal Ahire** (202301040067)
- **Rajveersinh Kher** (202301040233)
- **Atharva Suryawanshi** (202301040283)

---

## Assignment Details
- **Assignment:** LAB ASSIGNMENT 5: LSTM-Based AI Agent for Sequence Prediction

---

## Project Overview
This project implements an **LSTM-based Next Word Prediction System** trained on the Simple English Wikipedia dataset. The system is deployed using **FastAPI** on Render (cloud) and integrated into an **n8n AI Agent workflow** named **Adam** — creating a complete end-to-end industry-relevant AI system.

### Key Features
- **NLP Preprocessing Pipeline** — Lowercasing, special character removal, n-gram sequence generation, pre-padding
- **LSTM Sequence Learning** — Captures long-term dependencies in text for context-aware predictions
- **Memory-Efficient Training** — Uses sparse categorical crossentropy (avoids one-hot encoding RAM overhead)
- **Early Stopping + Checkpointing** — Saves best model weights, protects against Colab disconnects
- **FastAPI Deployment** — Two endpoints: `/predict` (next word) and `/generate` (sentence continuation)
- **n8n AI Agent (Adam)** — Public chat interface powered by the LSTM API
- **UptimeRobot Monitoring** — Keeps the Render service alive with health checks every 5 minutes

---

## Project Structure

```
Lab-Assignment-5-lstm-sequence-prediction/
│
├── LSTM_Text_Prediction.ipynb   # Main training notebook (run on Google Colab)
├── app.py                       # FastAPI deployment server
├── requirements.txt             # Python dependencies for Render
├── render.yaml                  # Render cloud deployment config
├── .python-version              # Python 3.11.0 (required for Keras 3.13.2)
├── README.md                    # This file
│
├── dataset/                     # Dataset folder
│   └── AllCombined.txt          # Simple English Wikipedia plain text
│
└── model/                       # Trained model files
    ├── lstm_model.keras         # Trained LSTM model
    ├── tokenizer.pkl            # Fitted Keras tokenizer
    └── max_seq_len.pkl          # Max sequence length (required by API)
```

---

## Dataset

| Property | Details |
|----------|---------|
| Name | Plain Text Wikipedia (Simple English) |
| Source | https://www.kaggle.com/datasets/josephrmartinez/simple-english-wikipedia |
| File | AllCombined.txt |
| Full Size | 249,396 articles, 31M tokens, ~400MB |
| Subset Used | Random 3,000 lines (seed=42 for reproducibility) |

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.19 / Keras 3.13.2 |
| API Framework | FastAPI + Uvicorn |
| Training Environment | Google Colab |
| Cloud Deployment | Render (free tier) |
| AI Agent Workflow | n8n Cloud |
| Uptime Monitoring | UptimeRobot |

---

## Architecture

```
User
  │
  ▼
n8n Chat Agent (Adam)
  │  Chat Trigger → HTTP Request → Respond to Webhook
  ▼
FastAPI on Render
  │  POST /generate
  ▼
LSTM Model
  │  Embedding → LSTM(150) → Dropout(0.2) → Dense(softmax)
  ▼
Predicted Text
```

---

## Live Deployments

| Service | URL |
|---------|-----|
| FastAPI (Render) | https://lstm-sequence-prediction-system.onrender.com |
| API Docs (Swagger) | https://lstm-sequence-prediction-system.onrender.com/docs |
| Adam Chat Agent (n8n) | https://parimalahire18.app.n8n.cloud/webhook/6b5ee29f-0e50-48e8-86bc-f98a124ba40c/chat |

---

## Step 1 — Train the Model on Google Colab

1. Place `AllCombined.txt` inside the `dataset/` folder on Google Drive at:
   `My Drive/Lab-Assignment-5-lstm-sequence-prediciton/dataset/AllCombined.txt`
2. Open `LSTM_Text_Prediction.ipynb` in Google Colab
3. Run all cells — it will:
   - Mount your Drive and load the dataset automatically
   - Randomly sample 3,000 lines (seed=42)
   - Clean and preprocess the text (lowercase, remove special chars)
   - Build n-gram sequences and pad them
   - Train the LSTM model (up to 80 epochs with early stopping, patience=10)
   - Save `model/lstm_model.keras`, `model/tokenizer.pkl`, `model/max_seq_len.pkl`
4. Download the 3 saved model files from Colab

---

## Step 2 — Run the API Locally

Place the downloaded `model/` folder in the project root, then:

```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

API will be live at `http://localhost:8000`

---

## Step 3 — Deploy on Render

1. Push the project to GitHub (with `model/` folder included)
2. Go to render.com → New → Web Service
3. Connect the GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Click Deploy

---

## Step 4 — n8n AI Agent (Adam)

The n8n workflow connects a chat interface directly to the FastAPI:

| Node | Type | Purpose |
|------|------|---------|
| When chat message received | Chat Trigger | Receives user input |
| HTTP Request | POST /generate | Calls FastAPI with user text |
| Respond to Webhook | Text Response | Returns prediction to chat |

**Adam's greeting:**
> "Hi! 👋 I'm Adam, your LSTM Text Prediction Agent. Send me any text and I'll predict the next word!"

**Example predictions:**
**Example predictions:**
- "adam zampa was" → "adam zampa was born in the"
- "how are you" → "how are you in the united"
- "Am I in" → "Am I in the united states"

---

## API Documentation

### Home
- **URL:** `GET /`
- **Response:** `{"message": "LSTM Next Word Prediction API is running!"}`

### Health Check
- **URL:** `HEAD /health`
- **Response:** 200 OK (used by UptimeRobot)

### Predict Next Word
- **URL:** `POST /predict`
- **Request:** `{ "text": "Am i in" }`
- **Response:** `{ "input": "Am I in", "predicted_next_word": "the" }`

### Generate Sentence
- **URL:** `POST /generate`
- **Request:** `{ "text": "Am I in", "num_words": 3 }`
- **Response:** `{ "input": "Am I in", "generated_text": "Am I in the united states" }`

---

## LSTM Mathematical Model

The LSTM cell manages information through three gates:

| Gate | Equation | Purpose |
|------|----------|---------|
| Forget Gate | f_t = sigmoid(W_f . [h_(t-1), x_t] + b_f) | Discard irrelevant past info |
| Input Gate | i_t = sigmoid(W_i . [h_(t-1), x_t] + b_i) | Select new info to store |
| Cell State | C_t = f_t * C_(t-1) + i_t * tanh(W_c . [h_(t-1), x_t] + b_c) | Long-term memory carrier |
| Output Gate | o_t = sigmoid(W_o . [h_(t-1), x_t] + b_o) | Control what to output |
| Hidden State | h_t = o_t * tanh(C_t) | Short-term output / next input |

---

## AI Acknowledgement *(Mandatory as per assignment)*

| Tool | Purpose | Sections Used |
|------|---------|---------------|
| **Claude (Anthropic)** | Reference and guidance during implementation | Some parts of API code and deployment configuration |

