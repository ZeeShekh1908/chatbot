import json
import os
import google.generativeai as genai
from typing import Dict, List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Initialize FastAPI app and Gemini Client ---
app = FastAPI()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # CORRECTED MODEL NAME
    model = genai.GenerativeModel('gemini-2.0-flash')

except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

# --- In-Memory Storage ---
conversation_history: Dict[str, List[Dict]] = {}
mock_data = json.load(open("data.json"))

# --- Prompt Engineering ---
SYSTEM_PROMPT = """
You are an expert, accessible AI helpdesk assistant. Your name is 'HelpBot'.
- Be friendly, concise, and helpful.
- Use the provided context data to answer user questions.
- Never make up information. If the data is not available, say so.
- When tracking an order, always state the order ID, status, and estimated delivery.
- Your primary goal is to help users efficiently.
"""

# --- Helper to format history for Gemini ---
def format_history_for_gemini(history: List[Dict]) -> List[Dict]:
    gemini_history = []
    for message in history:
        # Gemini uses 'model' for the assistant's role
        role = "model" if message["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [message["content"]]})
    return gemini_history

# --- Core Logic ---
@app.post("/chat")
async def chat(request: Request):
    if not model:
        return {"response": "The AI model is not configured correctly. Please check the API key."}
        
    data = await request.json()
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default_session")

    if not user_message:
        return {"response": "Sorry, I didn't receive a message."}

    # 1. Manage Conversation History
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    conversation_history[session_id].append({"role": "user", "content": user_message})

    # Check for simple fallback intent
    if "human" in user_message.lower() or "agent" in user_message.lower():
        response_text = "I understand. I am connecting you to a human agent now."
    else:
        try:
            # 2. THIS IS THE CORRECTED LOGIC
            context_data = f"CONTEXT: {json.dumps(mock_data)}"
            
            # The full history including the new instructions
            full_history = [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + context_data},
                {"role": "assistant", "content": "Understood. I am HelpBot and will follow these instructions."},
            ] + conversation_history[session_id]

            # Format for the API
            api_prompt = format_history_for_gemini(full_history)

            # 3. Call the Gemini API using the reliable generate_content method
            response = model.generate_content(api_prompt)
            response_text = response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            response_text = "I'm having trouble connecting to my brain right now. Please try again in a moment."

    # 4. Update history with the correct response
    conversation_history[session_id].append({"role": "assistant", "content": response_text})
    
    return {"response": response_text}

# --- Serve Frontend ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("../frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)