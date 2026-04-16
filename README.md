# Lab Assignment 5 – LSTM Model  
## Sequence Prediction System using LSTM with FastAPI Deployment  

---

## Course Details

**Course Name:** Deep Learning  
**Lab Assignment:** Lab Assignment 5  
**Semester:** VI  
**Topic:** LSTM-Based Sequence Prediction System with Deployment  

---

## Group Details

| Student Name | PRN |
|--------------|-------------------|
| Parimal Ahire | 202301040067 |
| Atharva Suryawanshi | 202301040283 |
| Rajveersinh Kher | 202301040233 |
| Mohit Patil | 202301040272 |

---

## Objective

The objective of this lab assignment is to design and implement an LSTM-based sequence prediction system capable of predicting the next word in a given text sequence. The project demonstrates the application of deep learning techniques in natural language processing and showcases real-time deployment using an API-based approach.

The following tasks are implemented:

- Text preprocessing and sequence generation  
- Next-word prediction using LSTM  
- Multi-word text generation  
- Model training and evaluation  
- Model saving and reuse  
- API-based deployment using FastAPI  
- Real-time prediction through REST endpoints  

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training the Model (Colab)](#training-the-model-colab)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [LSTM Architecture](#lstm-architecture)
- [AI Acknowledgement](#ai-acknowledgement)

---

## Project Overview

This system takes a sequence of words as input and predicts the most likely next word using a trained LSTM neural network.

**Tech Stack:**
- Python 3.10
- TensorFlow / Keras — LSTM model
- FastAPI — REST API deployment
- Google Colab — model training
- Kaggle — dataset source

---

## Dataset

| Property | Details |
|----------|---------|
| **Name** | Plain Text Wikipedia (Simple English) |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/josephrmartinez/simple-english-wikipedia) |
| **Full Size** | 249,396 articles, 31M tokens, ~400MB |
| **Subset Used** | 30,000 lines (`data/dataset_sample.txt`) |
| **Description** | Clean, simplified English Wikipedia articles stripped of all formatting |

**Preprocessing Steps:**
1. Filtered lines shorter than 20 characters (removes titles and blank lines)
2. Converted all text to lowercase
3. Removed special characters and punctuation
4. Tokenized sentences into word sequences
5. Generated n-gram input-output pairs
6. Applied pre-padding to normalize sequence length

---

## Project Structure

```
lstm-sequence-prediction/
│
├── notebook/
│   └── LSTM_Text_Prediction.ipynb   # Colab training notebook (mandatory)
│
├── model/                           # Generated after training
│   ├── lstm_model.h5                # Trained LSTM model
│   ├── tokenizer.pkl                # Fitted Keras tokenizer
│   └── max_seq_len.pkl              # Max sequence length
│
├── api/
│   └── app.py                       # FastAPI deployment
│
├── data/
│   ├── AllCombined.txt              # Full Kaggle dataset (not committed)
│   └── dataset_sample.txt           # 3,000 line subset used for training
│
├── screenshots/
│   └── api_testing.png              # Swagger UI test proof
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/lstm-sequence-prediction.git
cd lstm-sequence-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download model files from Colab
After training in Colab, download and place in `model/`:
- `lstm_model.h5`
- `tokenizer.pkl`
- `max_seq_len.pkl`

---

## Training the Model (Colab)

1. Open `notebook/LSTM_Text_Prediction.ipynb` in Google Colab
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Run all cells — upload `data/dataset_sample.txt` when prompted
4. After training completes, download the 3 files from `model/`
5. Place them in your local `model/` folder

**Expected training time:** ~10–15 minutes on Colab GPU

---

## Running the API

```bash
uvicorn api.app:app --reload
```

API will be live at: `http://127.0.0.1:8000`

Swagger UI (interactive docs): `http://127.0.0.1:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/predict?text=...` | Predict next single word |
| GET | `/generate?text=...&num_words=5` | Generate multiple next words |
| GET | `/vocab` | Vocabulary info |

### Example Requests

**Predict next word:**
```
GET /predict?text=machine learning is
```
```json
{
  "input": "machine learning is",
  "next_word": "a",
  "full_text": "machine learning is a"
}
```

**Generate sentence:**
```
GET /generate?text=deep learning&num_words=4
```
```json
{
  "input": "deep learning",
  "generated": "deep learning uses neural networks with",
  "words_added": ["uses", "neural", "networks", "with"],
  "count": 4
}
```

---

## Testing

1. Run the API with `uvicorn api.app:app --reload`
2. Open `http://127.0.0.1:8000/docs`
3. Test the `/predict` endpoint via Swagger UI
4. Take a screenshot and save to `screenshots/api_testing.png`

---

## LSTM Architecture

```
Input Sequence
      ↓
Embedding Layer (vocab_size → 64 dims)
      ↓
LSTM Layer (150 units)
      ↓
Dropout (0.2)
      ↓
Dense Layer (softmax → vocab_size)
      ↓
Predicted Next Word
```

**Key equations:**

| Gate | Formula | Purpose |
|------|---------|---------|
| Forget | f_t = σ(W_f · [h_(t-1), x_t] + b_f) | Discard old info |
| Input | i_t = σ(W_i · [h_(t-1), x_t] + b_i) | Store new info |
| Cell | C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t | Update memory |
| Output | o_t = σ(W_o · [h_(t-1), x_t] + b_o) | Control output |
| Hidden | h_t = o_t ⊙ tanh(C_t) | Final output |

---

## AI Acknowledgement

As required by the assignment:

| Tool | Purpose | Sections |
|------|---------|---------|
| **Claude (Anthropic)** | Code structuring, notebook layout, FastAPI scaffolding | Dataset loading, model architecture, API endpoints, README |

All LSTM concepts, mathematical understanding, and design decisions were reviewed and understood by all group members. AI was used as a coding assistant to accelerate development.

