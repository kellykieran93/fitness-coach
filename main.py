import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from mem0 import MemoryClient

app = FastAPI(title="Personalized Fitness Coach API")

# Initialize API clients from environment variables
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
mem0_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 1. Retrieve relevant memories using updated Mem0 filters syntax
        memory_response = mem0_client.search(
            query=request.message, 
            filters={"user_id": request.user_id}
        )
        
        # Format extracted facts safely regardless of response format
        if isinstance(memory_response, dict):
            memories_list = memory_response.get("results", [])
        elif isinstance(memory_response, list):
            memories_list = memory_response
        else:
            memories_list = []
            
        facts = [m["memory"] for m in memories_list if isinstance(m, dict) and "memory" in m]
        memory_context = "\n- ".join(facts) if facts else "No prior history recorded yet."

        # 2. Build system prompt with retrieved long-term memory context
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
                {"role": "user", "content": request.message}
            ],
            temperature=0.7,
        )
        bot_response = completion.choices[0].message.content

        # 4. Save message pair to Mem0
        messages_to_add = [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": bot_response}
        ]
        mem0_client.add(messages=messages_to_add, user_id=request.user_id)

        return {"response": bot_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
