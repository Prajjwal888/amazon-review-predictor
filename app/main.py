from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from huggingface_hub import login
import os

app = FastAPI()

login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

classifier = pipeline("text-classification", model="prajjwal888/roberta-finetuned-review-classifier")

class ReviewInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(review: ReviewInput):
    prediction = classifier(review.text)
    return {"result": prediction}
