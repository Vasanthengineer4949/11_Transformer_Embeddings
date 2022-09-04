from array import array
from pyexpat import model
from fastapi import FastAPI
import joblib
import uvicorn
import numpy as np
from sentence_transformers import SentenceTransformer
import config

app = FastAPI(debug=True)

@app.get('/')
def home():
    return {'Project': 'Transformer Embedding ML Classifier'}

@app.get('/predict')
def predict(sentence_str:str):

    pred_model1 = joblib.load('artifacts/model_out/cat_imb.pkl')
    pred_model2 = joblib.load('artifacts/model_out/logreg_imb.pkl')
    embed = SentenceTransformer(config.MODEL_CKPT)
    inp = embed.encode(sentence_str)
    model1pred = np.array(pred_model1.predict_proba([inp]))
    model2pred = np.array(pred_model2.predict_proba([inp]))
    combinedmodelpred = np.array((model1pred+model2pred)/2)
    combinedmodelpred = combinedmodelpred.tolist()
    combinedmodelpred = combinedmodelpred[0]

    return combinedmodelpred.index(max(combinedmodelpred))
    
    # return {f"The predicted score is {round(prediction[0])}"}


if __name__=="__main__":
    uvicorn.run(app)