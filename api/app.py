import numpy as np
import pickle
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="LSTM Text Prediction API",
    description="Predicts the next word in a sequence using a trained LSTM model.",
    version="1.0.0"
)

# Enable CORS (for testing / frontend use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Paths (FIXED)
# ─────────────────────────────────────────────
MODEL_PATH     = "model/lstm_model.keras"   # ✅ FIXED
TOKENIZER_PATH = "model/tokenizer.pkl"
MAXLEN_PATH    = "model/max_seq_len.pkl"

# ─────────────────────────────────────────────
#  Validate Files
# ─────────────────────────────────────────────
for path in [MODEL_PATH, TOKENIZER_PATH, MAXLEN_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Make sure you downloaded files from Colab into /model folder."
        )

# ─────────────────────────────────────────────
#  Load Model & Artifacts
# ─────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)

print("Loading tokenizer...")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("Loading max sequence length...")
with open(MAXLEN_PATH, "rb") as f:
    max_seq_len = pickle.load(f)

print(f"✅ Model loaded | Vocab size: {len(tokenizer.word_index)} | Max len: {max_seq_len}")

# ─────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────
def predict_next_word(seed_text: str) -> str:
    seed_text = seed_text.lower().strip()

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    if not token_list:
        raise ValueError("Input text not in vocabulary.")

    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')

    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = int(np.argmax(predicted_probs))

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "[unknown]"


def generate_sequence(seed_text: str, num_words: int) -> str:
    result = seed_text.lower().strip()

    for _ in range(num_words):
        next_word = predict_next_word(result)

        if next_word == "[unknown]":
            break

        result += " " + next_word

    return result

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "LSTM Text Prediction API is live 🚀",
        "endpoints": {
            "predict": "/predict?text=machine+learning+is",
            "generate": "/generate?text=deep+learning&num_words=5",
            "docs": "/docs"
        }
    }


@app.get("/predict")
def predict(text: str):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        next_word = predict_next_word(text)
        return {
            "input": text,
            "next_word": next_word,
            "full_text": text + " " + next_word
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/generate")
def generate(text: str, num_words: int = 5):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if num_words < 1 or num_words > 20:
        raise HTTPException(status_code=400, detail="num_words must be between 1 and 20.")

    try:
        generated = generate_sequence(text, num_words)

        words_added = generated[len(text):].strip().split()

        return {
            "input": text,
            "generated": generated,
            "words_added": words_added,
            "count": len(words_added)
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/vocab")
def vocab():
    words = list(tokenizer.word_index.keys())

    return {
        "vocab_size": len(tokenizer.word_index),
        "max_sequence_length": max_seq_len,
        "sample_words": words[:30]
    }