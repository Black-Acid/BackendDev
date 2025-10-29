# from fastapi import Depends, FastAPI, HTTPException, security
# import schemas as sma
# from sqlalchemy import orm 
# import services as sv

# import os
# from dotenv import load_dotenv
# import redis 
# import json


# load_dotenv()

# redisClient = redis.Redis(host='localhost', port=6379, decode_responses=True)




# app = FastAPI()

# @app.post("/api/register-user/")
# async def register_user(user: sma.UserRequest, db: orm.Session = Depends(sv.get_db) ):
#     checked_user = await sv.get_user(user, db)
#     if checked_user:
#         raise HTTPException(status_code=400, detail="A User already exists with that username")
#     try:
#         creating_user = await sv.create_user(user, db)
#     except:
#         raise HTTPException(status_code=400, detail="You made a bad request")
    
#     print(creating_user)
    
#     return await sv.create_token(creating_user)


# @app.post("/api/login")
# async def login(form_data: security.OAuth2PasswordRequestForm = Depends(), db: orm.Session = Depends(sv.get_db)):
#     db_user = await sv.login(form_data.username, form_data.password, db)

#     if not db_user:
#         raise HTTPException(status_code=401, detail="Invalid credentials")

#     return await sv.create_token(db_user)


# @app.post("/api/send-data")
# async def query(
#     data: sma.UserQuery, 
#     user: sma.UserResponse = Depends(sv.get_current_user), 
#     db: orm.Session = Depends(sv.get_db)
# ):
    
#     hashset_name = "chat_responses"
    
    
#     #check redis to see if we have a question like that
#     cache_response = redisClient.hget(hashset_name, data.message)
    
#     # return the response if there is one
#     if cache_response:
#         print("I got it from cache")
#         return cache_response
    
#     else:
        
#         data_to_push = sma.RabbitMQPush(id=user.id, message=data.message)
#         sv.push_to_rabbitmq(data_to_push)
        
#         response = await sv.get_query()
#         print(response)
#         redisClient.hset(hashset_name, data.message, json.dumps(response))
#         return response
        
        
        
        

        
#         # # fetch the response from groq ai and save it into redis and display it
#         # print("Naaa I got it from groq directly")
        
        
#         # #Convert the recieved data to an instance of the UserQueryResponse Schema
#         # queryResponse = sma.UserQueryResponse(message=data.message, response=groq_response)
#         # await sv.save_queries(queryResponse, db, user.id)

#         # redisClient.hset(hashset_name, data.message, groq_response)
        
#         # return groq_response



from fastapi import Depends, FastAPI, HTTPException, security
import schemas as sma
from sqlalchemy import orm
import os
from dotenv import load_dotenv
import redis
import json

load_dotenv()
import services as sv

redisClient = redis.Redis(host='localhost', port=6379, decode_responses=True)

app = FastAPI()

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


@app.post("/api/send-data")
async def query(
    data: sma.UserQuery,
    user: sma.UserResponse = Depends(sv.get_current_user),
    db: orm.Session = Depends(sv.get_db)
):
    hashset_name = "chat_responses"

    # Check Redis cache
    cache_response = redisClient.hget(hashset_name, data.message)
    if cache_response:
        print("I got it from cache")
        return json.loads(cache_response)

    # If not in cache, fetch response directly from Groq AI (or whatever service)
    print("Fetching new response from Groq or service...")
    response = await sv.get_response2(data.message)

    # Save to Redis
    redisClient.hset(hashset_name, data.message, json.dumps(response))

    # Save query in DB (optional, for tracking)
    queryResponse = sma.UserQueryResponse(message=data.message, response=response)
    await sv.save_queries(queryResponse, db, user.id)

    return response
