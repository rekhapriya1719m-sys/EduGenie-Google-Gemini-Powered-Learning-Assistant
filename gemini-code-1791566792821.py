import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is missing!")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="EduGenie API",
    description="Google Gemini Powered Learning Assistant Backend",
    version="1.0.0"
)

# Setup Templates Directory
templates = Jinja2Templates(directory="templates")

# Initialize Gemini Model
model = genai.GenerativeModel("gemini-1.5-flash")

# Request Models
class AskQuestionRequest(BaseModel):
    prompt: str

class GenerateQuizRequest(BaseModel):
    topic: str
    num_questions: int = 3

class SummarizeRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """Serve the main web dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/ask")
async def ask_tutor(data: AskQuestionRequest):
    """
    Acts as an interactive AI Tutor providing step-by-step educational answers.
    """
    try:
        system_instruction = (
            "You are EduGenie, an encouraging, step-by-step academic learning assistant. "
            "Explain concepts simply, use clear analogies, and ask a follow-up question to check understanding."
        )
        full_prompt = f"{system_instruction}\n\nUser Question: {data.prompt}"
        
        response = model.generate_content(full_prompt)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quiz")
async def generate_quiz(data: GenerateQuizRequest):
    """
    Generates multiple-choice quiz questions for a given topic.
    """
    try:
        prompt = (
            f"Generate a {data.num_questions}-question multiple-choice quiz on the topic '{data.topic}'. "
            "Format each question clearly with options (A, B, C, D) and specify the correct answer at the end of each question."
        )
        response = model.generate_content(prompt)
        return {"topic": data.topic, "quiz": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize")
async def summarize_notes(data: SummarizeRequest):
    """
    Summarizes long study notes into concise bullet points.
    """
    try:
        prompt = (
            "You are EduGenie's note summarizer. Summarize the following text into concise bullet points "
            "and list key takeaways:\n\n" + data.text
        )
        response = model.generate_content(prompt)
        return {"summary": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)