from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import uvicorn

app = FastAPI(title='LSTM Next Word Prediction API')

@app.get('/')
def home():
    return {
        'message': 'LSTM Next Word Prediction API is running!',
        'instructions': 'Use /predict for next word, /generate for sentence continuation, or visit /docs for interactive documentation.'
    }

# Load model, tokenizer and max_seq_len
model = tf.keras.models.load_model('model/lstm_model.keras')

with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('model/max_seq_len.pkl', 'rb') as f:
    max_seq_len = pickle.load(f)


class PredictionRequest(BaseModel):
    text: str

class GenerateRequest(BaseModel):
    text: str
    num_words: int = 3


def predict_next_word(seed_text):
    seed_text = seed_text.lower()
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return "[unknown]"


@app.post('/predict')
def predict(request: PredictionRequest):
    predicted_word = predict_next_word(request.text)
    return {
        'input': request.text,
        'predicted_next_word': predicted_word
    }


@app.post('/generate')
def generate(request: GenerateRequest):
    result = request.text
    for _ in range(request.num_words):
        next_word = predict_next_word(result)
        result += ' ' + next_word
    return {
        'input': request.text,
        'generated_text': result
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
