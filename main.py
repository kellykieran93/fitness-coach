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
    system_prompt = f"""You are an elite, empathetic health and fitness coach with real-time location and web search capabilities.
Your goal is to provide safe, highly personalized workout advice, meal planning, and outdoor recreation recommendations (hikes, parks, trail runs).

WHAT YOU KNOW ABOUT THIS USER FROM PAST SESSIONS:
- {memory_context}

Instructions:
- Use the facts above to tailor your advice (e.g., respect injuries, food allergies, and preferences).
- If the user asks for hikes, parks, or local outdoor spots, perform a live search to give real, accurate location details and safety tips.
- Keep responses concise and practical.
"""

    # 3. Request LLM completion from Groq with live browser search tool enabled
    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        tools=[{"type": "browser_search"}],
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
# Helper to extract nutrition macros from image URL
async def analyze_food_image(image_url: str) -> str:
    completion = groq_client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Analyze this food packaging or nutrition label. Extract the food item name, serving size, calories, protein, carbs, and fat content. Keep the summary short and formatted as bullet points."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content
# Telegram Webhook endpoint supporting photos and text
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
        if "message" in data:
            message = data["message"]
            chat_id = str(message["chat"]["id"])

            # 1. Handle Photo Uploads
            if "photo" in message:
                file_id = message["photo"][-1]["file_id"]
                
                async with httpx.AsyncClient() as client:
                    file_info = await client.get(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
                    )
                    file_path = file_info.json()["result"]["file_path"]
                    image_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"

                    macro_summary = await analyze_food_image(image_url)

                    mem0_client.add(
                        messages=[
                            {"role": "user", "content": "I ate/logged this food label photo."},
                            {"role": "assistant", "content": macro_summary}
                        ],
                        user_id=chat_id
                    )

                    reply_text = f"📸 **Nutrition Label Processed & Saved!**\n\n{macro_summary}"
                    await client.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                        json={"chat_id": chat_id, "text": reply_text}
                    )

            # 2. Handle Regular Text Messages
            elif "text" in message:
                user_text = message["text"]
                bot_response = await get_coach_response(chat_id, user_text)

                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                        json={"chat_id": chat_id, "text": bot_response}
                    )

        return {"status": "ok"}
    except Exception as e:
        print(f"Webhook error: {e}")
        return {"status": "ok"}
