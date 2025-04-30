from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from main import initialize_rag
from lightrag import QueryParam
import traceback
import json

app = FastAPI()

class UserRequest(BaseModel):
    persona: object;
    user_id: str

rag = None

@app.on_event("startup")
async def on_startup():
    global rag
    rag = await initialize_rag()

@app.post("/match-user")
async def match_user(data: UserRequest):
    try:
        # question = (
        #     f"You will receive a current user's persona and a list of other users' personas. "
        #     f"For now, let's take user = {data.user_id}. consider the following persona object: \n{data.persona}\n "
        #     "Your task is to find the top 5 matches based on shared age range, goals, interests, hobbies. "
        #     "For each match, explain the reason for the match based on persona similarities."
        #     "!!!IMPORTANT!!!: the response should be an array of the `_id` from the database of the matched users objects. the format of the response should be ONLY the following and NOTHING else:  [{'user1._id': 'user1._id', 'user1.name': 'user1.name'}, {'user2._id': 'user2._id', 'user2.name': 'user2.name'}, {'user3._id': 'user3._id', 'user3.name': 'user3.name'}, {'user4._id': 'user4._id', 'user4.name': 'user4.name'}, {'user5._id': 'user5._id', 'user5.name': 'user5.name'} ] \n The _id should be a string taken from the database, the same _id used to retrieve the user in the database, and the name should be a string."
        # )
        question = f"""
You are helping match users based on their persona information.

Current user's persona:
{json.dumps(data.persona, indent=2)}

Your task:
- Find the top 5 users that best match the persona based on shared age range, goals, interests, hobbies, or other meaningful factors.
- Each user document contains a MongoDB _id field (e.g., "_id": "65f09a1a82e2c934dc4f6f82") and a username field (e.g., "username": "John Doe").

Rules:
- Retrieve and output ONLY the _id and username fields from the matching user documents. (Example: "67feed453ac8038038394731" and NOT: "0", "1", "2" etc.")
- DO NOT invent or guess IDs. ONLY use the existing _id field from the user document.
- If fewer than 5 matches exist, return as many as available.
- Do not include explanations, descriptions, or any other text outside the JSON.
- Ensure the MongoDB user IDs are correctly taken from the database and are valid.
- If there are fewer than 5 matches, return as many as available following the same format. Avoid repeating the same user in the array more
- DO NOT include the same user of the user_id provided by the request in the matches array.

Output:
Return ONLY a valid JSON object using the following structure, with no extra text, symbols etc.:
{{"matches":[{{"_id":"user_id", "username":"user_name"}}, {{"_id":"user_id", "username":"user_name"}}, {{"_id":"user_id", "username":"user_name"}}, {{"_id":"user_id", "username":"user_name"}}, {{"_id":"user_id", "username":"user_name"}}]}}
        """
        print("QUESTION: \n " + question)
        result = await rag.aquery(query=question, param=QueryParam(mode="local"))
        return { "matches": result }
    except Exception as e:
        print("Error")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
