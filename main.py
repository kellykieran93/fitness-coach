import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from groq import Groq
from mem0 import MemoryClient
from mem0 import MemoryClient
import sqlite3
from datetime import date

app = FastAPI(title="Personalized Fitness Coach API")

# Initialize SQLite database for daily macros
def init_db():
    conn = sqlite3.connect("macros.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS food_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            log_date TEXT,
            food_name TEXT,
            calories INTEGER,
            protein REAL,
            carbs REAL,
            fat REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Helper to fetch today's total macros for a user
def get_daily_macros(user_id: str) -> str:
    conn = sqlite3.connect("macros.db")
    cursor = conn.cursor()
    today = str(date.today())
    cursor.execute("""
        SELECT SUM(calories), SUM(protein), SUM(carbs), SUM(fat), GROUP_CONCAT(food_name, ', ')
        FROM food_logs WHERE user_id = ? AND log_date = ?
    """, (user_id, today))
    result = cursor.fetchone()
    conn.close()

    if not result or result[0] is None:
        return "No macros logged for today yet!"
    
    cal, pro, carb, fat, items = result
    return f"""📊 **Today's Macro Summary ({today}):**
- **Items Logged:** {items}
- **Calories:** {cal or 0} kcal
- **Protein:** {pro or 0}g
- **Carbs:** {carb or 0}g
- **Fat:** {fat or 0}g"""
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
Your goal is to provide safe, highly personalized workout advice, meal planning, and outdoor recreation recommendations.

WHAT YOU KNOW ABOUT THIS USER FROM PAST SESSIONS:
- {memory_context}

ALLTRAILS LOCATION INSTRUCTIONS:
- Whenever the user asks for hikes, trail runs, or outdoor walks, prioritize findings specifically from AllTrails (alltrails.com).
- Use live browser search to locate actual AllTrails links for each trail.
- Format trail outputs cleanly using Telegram Markdown:
  * **[Trail Name](AllTrails URL)**
  * **Distance & Difficulty**: e.g., 3.2 miles | Easy
  * **Trail Type**: Loop/Out & Back
  * **Why it fits**: Explain how it respects user preferences or injuries (e.g., flat surface to avoid slips).
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
    # Helper to convert Telegram voice note (.ogg) to text using Groq Whisper
async def transcribe_telegram_voice(file_id: str) -> str:
    async with httpx.AsyncClient() as client:
        # 1. Get file path from Telegram
        file_info = await client.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile?file_id={file_id}"
        )
        file_path = file_info.json()["result"]["file_path"]
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"

        # 2. Download audio bytes from Telegram
        audio_response = await client.get(file_url)
        audio_bytes = audio_response.content

    # 3. Pass audio bytes directly to Groq Whisper API
    transcription = groq_client.audio.transcriptions.create(
        file=("voice.ogg", audio_bytes),
        model="whisper-large-v3",
        prompt="Fitness and nutrition context",
        response_format="json",
        language="en"
    )
    return transcription.text
# Telegram Webhook endpoint supporting Voice, Photos, and Text
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
        if "message" in data:
            message = data["message"]
            chat_id = str(message["chat"]["id"])

            # 1. Handle Voice Notes
            if "voice" in message or "audio" in message:
                voice_data = message.get("voice") or message.get("audio")
                file_id = voice_data["file_id"]

                # Transcribe audio using Whisper
                transcribed_text = await transcribe_telegram_voice(file_id)

                # Process transcribed text through core AI + Mem0 memory engine
                bot_response = await get_coach_response(chat_id, transcribed_text)

                # Send response back to Telegram
                reply_text = f"🎙️ *You said:* \"{transcribed_text}\"\n\n{bot_response}"
                
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": reply_text,
                            "parse_mode": "Markdown",
                            "disable_web_page_preview": False
                        }
                    )

            # 2. Handle Photo Uploads
            elif "photo" in message:
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
                        json={"chat_id": chat_id, "text": reply_text, "parse_mode": "Markdown"}
                    )

          # 3. Handle Regular Text Messages & Commands
            elif "text" in message:
                user_text = message["text"].strip()

                # Check if user wants to view macros summary
                if user_text.lower() in ["/macros", "/today"]:
                    reply_text = get_daily_macros(chat_id)
                
                # Check if user wants to wipe memory
                elif user_text.lower() in ["/reset", "/clear"]:
                    try:
                        mem0_client.delete_all(user_id=chat_id)
                        reply_text = "🧹 **Memory Wiped!** Let's start fresh."
                    except Exception as err:
                        reply_text = f"Failed to clear memory: {err}"
                
                else:
                    # Regular AI coach query
                    reply_text = await get_coach_response(chat_id, user_text)

                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                        json={
                            "chat_id": chat_id, 
                            "text": reply_text,
                            "parse_mode": "Markdown",
                            "disable_web_page_preview": False
                        }
                    )
        return {"status": "ok"}
    except Exception as e:
        print(f"Webhook error: {e}")
        return {"status": "ok"}
