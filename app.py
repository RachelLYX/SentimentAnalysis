from flask import Flask, render_template, request
import joblib
import os
from openai import OpenAI
import time

app = Flask(__name__)

model = joblib.load("models/best_sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

client = OpenAI()

def genai_sentiment(review_text, max_retries=3):
    """
    Call OpenAI's GPT-4o-mini to classify sentiment.
    Retries if rate limit is hit.
    """

    prompt = f"""
    You are an AI Assistant for analysing product reviews.

    Given the review below, do the following:
    1. Summarise the review.
    2. Classify the sentiment as Positive, Neutral, or Negative.
    3. Generate a short, empathetic business response.
    4. End by asking the customer what action they would like the business to take 
       (e.g., refund, replacement, further support, or feedback)

    IMPORTANT: Do NOT use any markdown formatting (no **, *, _, etc.). Output plain text only.

    Review:
    {review_text}

    Output format:
    Summary:
    Sentiment:
    Business Response (with a question to the customer):
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                temperature=0
            )
            return response.output_text.strip()
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries:
                wait_time = 60 * attempt
                print(f"Rate limit hit. Waiting {wait_time}s before retrying...")
                time.sleep(60)
            else:
                print(f"GenAI API error: {e}")
                return f"GenAI error: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    ml_result = None
    genai_result = None
    text = ""

    if request.method == "POST":
        text = request.form.get("review", "")

        X = vectorizer.transform([text])
        ml_result = model.predict(X)[0]

        try:
            genai_result = genai_sentiment(text)
            genai_result = genai_result.replace("\n", "<br>")
        except Exception as e:
            genai_result = f"GenAI error: {e}"
    
    return render_template(
        "index.html",
        ml_result=ml_result,
        genai_result=genai_result,
        text = text
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

