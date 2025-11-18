

from fastapi import Depends, FastAPI, HTTPException, security
import schemas as sma
from sqlalchemy import orm
import os
from dotenv import load_dotenv
import redis
import json
import uvicorn
from contextlib import asynccontextmanager

load_dotenv()
import services as sv



# redisClient = redis.Redis(host='localhost', port=6379, decode_responses=True)



    
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs at startup
    sv.initialize_services()
    yield
    # Here you could add shutdown logic if needed

app = FastAPI(lifespan=lifespan)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

@app.post("/api/register-user/")
async def register_user(user: sma.UserRequest, db: orm.Session = Depends(sv.get_db)):
    checked_user = await sv.get_user(user, db)
    if checked_user:
        raise HTTPException(status_code=400, detail="A User already exists with that username")
    try:
        creating_user = await sv.create_user(user, db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad request: {e}")

    print(creating_user)
    return await sv.create_token(creating_user)


@app.post("/api/login")
async def login(form_data: security.OAuth2PasswordRequestForm = Depends(), db: orm.Session = Depends(sv.get_db)):
    db_user = await sv.login(form_data.username, form_data.password, db)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return await sv.create_token(db_user)


# @app.post("/api/send-data")
# async def query(
#     data: sma.UserQuery,
#     user: sma.UserResponse = Depends(sv.get_current_user),
#     db: orm.Session = Depends(sv.get_db)
# ):
#     hashset_name = "chat_responses"

#     # Check Redis cache
#     cache_response = redisClient.hget(hashset_name, data.message)
#     if cache_response:
#         print("I got it from cache")
#         return json.loads(cache_response)

#     # If not in cache, fetch response directly from Groq AI (or whatever service)
#     print("Fetching new response from Groq or service...")
#     response = await sv.get_response2(data.message)

#     # Save to Redis
#     redisClient.hset(hashset_name, data.message, json.dumps(response))

#     # Save query in DB (optional, for tracking)
#     queryResponse = sma.UserQueryResponse(message=data.message, response=response)
#     await sv.save_queries(queryResponse, db, user.id)

#     return response


@app.post("/api/send-data")
async def query(
    data: sma.UserQuery,
    user: sma.UserResponse = Depends(sv.get_current_user),
    db: orm.Session = Depends(sv.get_db)
):
    hashset_name = "chat_responses"
    user_id = str(user.id)

    # If user is already in a question session, handle conversational logic
    if user_id in sv.user_sessions:
        response = await sv.handle_conversation(user_id, data.message, sv.retriever)
        return {"response": response}

    # If it's a new message (no session yet), check cache first
    # cache_response = redisClient.hget(hashset_name, data.message)
    # if cache_response:
    #     print("I got it from cache")
    #     return json.loads(cache_response)

    # Otherwise, start a new conversational flow
    print("Fetching new response from Groq or retriever...")
    response = await sv.handle_conversation(user_id, data.message, sv.retriever)

    # Save to Redis
    # redisClient.hset(hashset_name, data.message, json.dumps({"response": response}))

    # Save query in DB (optional)
    queryResponse = sma.UserQueryResponse(message=data.message, response=response)
    await sv.save_queries(queryResponse, db, user.id)

    return {"response": response}
