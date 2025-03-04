from fastapi import Depends, FastAPI, HTTPException, security
import schemas as sma
from sqlalchemy import orm 
import services as sv
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)


app = FastAPI()

@app.post("/api/register-user/")
async def register_user(user: sma.UserRequest, db: orm.Session = Depends(sv.get_db) ):
    checked_user = await sv.get_user(user, db)
    if checked_user:
        raise HTTPException(status_code=400, detail="A User already exists with that username")
    try:
        creating_user = await sv.create_user(user, db)
    except:
        raise HTTPException(status_code=400, detail="You made a bad request")
    
    print(creating_user)
    
    return await sv.create_token(creating_user)


@app.post("/api/login")
async def login(form_data: security.OAuth2PasswordRequestForm = Depends(), db: orm.Session = Depends(sv.get_db)):
    db_user = await sv.login(form_data.username, form_data.password, db)

    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return await sv.create_token(db_user)


@app.post("/api/send-data")
async def query(data: sma.UserQuery, user: sma.UserRequest = Depends(sv.get_current_user)):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": data.message
            }
        ],
        model = "llama-3.3-70b-versatile",
        stop=None,
        stream=False
    )
    
    return (chat_completion.choices[0].message.content)
