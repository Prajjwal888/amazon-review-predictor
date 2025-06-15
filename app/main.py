from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Review Classifier API")

# Load the model once on startup
classifier = pipeline("text-classification", model="prajjwal888/review-classifier")

class ReviewInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(review: ReviewInput):
    prediction = classifier(review.text)
    return {"result": prediction}
