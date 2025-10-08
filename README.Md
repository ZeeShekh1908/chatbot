# HelpBot – AI Helpdesk Chatbot

**HelpBot** is a FastAPI-based AI chatbot powered by Google Gemini 2.0.  
It provides friendly, context-aware support using a local knowledge base (`data.json`) and can track orders, answer queries, and escalate to a human agent when needed.

---

## 🛠 Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ZeeShekh1908/chatbot.git
   cd chatbot
Create and activate a virtual environment

bash
Copy code
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
Install dependencies

bash
Copy code
pip install -r requirements.txt
Set up environment variables
Create a .env file in the project root:

env
Copy code
GEMINI_API_KEY=your_ai_studio_api_key_here
Run the FastAPI server

bash
Copy code
uvicorn main:app --reload
Access the frontend at http://127.0.0.1:8000

🏛 Architecture Overview
text
Copy code
+------------------+
|  Frontend (HTML) |
+--------+---------+
         |
         v
+------------------+
| FastAPI Backend  |
|  - /chat endpoint|
|  - conversation  |
|    history       |
+--------+---------+
         |
         v
+------------------+
|  Gemini 2.0 AI   |
|  - generate_content |
|  - uses mock_data  |
+------------------+
Frontend sends user messages to /chat.

Backend maintains session-based history and applies system prompts.

Gemini 2.0 generates context-aware responses using data.json.

💬 Example Conversation
User:

Can you track my order #102?

HelpBot:

Your order #102 is currently in transit and is expected to arrive by Oct 12, 2025.

User:

I want to speak to a human.

HelpBot:

I understand. I am connecting you to a human agent now.

-> Accessibility Design Notes

Frontend HTML is screen-reader friendly with semantic tags.

Color contrasts are tested for visual accessibility.

Chat input supports keyboard navigation.

Messages are concise and structured for clarity.

⚠️ Known Limitations

AI responses are limited to the data.json context; it will not make up information.

In-memory conversation history is lost on server restart.

Currently only supports text-based chat; no multimedia input/output.

Relies on Google Gemini 2.0 API — downtime or rate limits may affect responses.

Escalation to human agents is simulated; real routing is not implemented.