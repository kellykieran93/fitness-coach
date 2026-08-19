import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from groq import Groq
from mem0 import MemoryClient

app = FastAPI(title="Personalized Fitness Coach API")

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
mem0_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

class ChatRequest(BaseModel):
    user_id: str
    message: str

# Core AI + Memory function shared by ReqBin and Telegram
async def get_coach_response(user_id: str, message: str) -> str:
    # 1. Retrieve user memories
    memory_response = mem0_client.search(
        query=message, 
        filters={"user_id": user_id}
    )
    
    if isinstance(memory_response, dict):
        memories_list = memory_response.get("results", [])
    elif isinstance(memory_response, list):
        memories_list = memory_response
    else:
        memories_list = []
        
    facts = [m["memory"] for m in memories_list if isinstance(m, dict) and "memory" in m]
    memory_context = "\n- ".join(facts) if facts else "No prior history recorded yet."

    # 2. Build system prompt
    system_prompt = f"""You are an elite, empathetic health and fitness coach.
Your goal is to provide safe, highly personalized workout and nutrition advice.

WHAT YOU KNOW ABOUT THIS USER FROM PAST SESSIONS:
- {memory_context}

Instructions:
- Use the facts above to tailor your advice (e.g., respect injuries, food allergies, and goals).
- Keep responses concise and practical.
"""

    # 3. Request LLM completion from Groq
    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        temperature=0.7,
    )
    bot_response = completion.choices[0].message.content

    # 4. Save new session memory
    mem0_client.add(
        messages=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": bot_response}
        ],
        user_id=user_id
    )

    return bot_response

# Original API route for ReqBin / Web testing
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await get_coach_response(request.user_id, request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Telegram Webhook endpoint
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
        if "message" in data and "text" in data["message"]:
            chat_id = str(data["message"]["chat"]["id"])
            user_text = data["message"]["text"]

            # Generate AI response
            bot_response = await get_coach_response(chat_id, user_text)

            # Post response back to Telegram
            if TELEGRAM_BOT_TOKEN:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                        json={"chat_id": chat_id, "text": bot_response}
                    )
        return {"status": "ok"}
    except Exception as e:
        print(f"Webhook error: {e}")
        return {"status": "ok"}
