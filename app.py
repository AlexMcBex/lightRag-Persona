from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from main import initialize_rag
from lightrag import QueryParam

app = FastAPI()

class UserRequest(BaseModel):
    user_id: str

rag = None

@app.on_event("startup")
async def on_startup():
    global rag
    rag = await initialize_rag()

@app.post("/match-user")
async def match_user(data: UserRequest):
    try:
        question = (
            f"You will receive a current user's persona and a list of other users' personas. "
            f"For now, let's take user = {data.user_id}. "
            "Your task is to find the top 5 matches based on shared age range, goals, interests, hobbies. "
            "For each match, explain the reason for the match based on persona similarities."
        )
        result = await rag.aquery(query=question, param=QueryParam(mode="hybrid"))
        return { "matches": result }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
