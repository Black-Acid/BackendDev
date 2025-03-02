from fastapi import Depends, FastAPI, HTTPException, security
import schemas as sma
from sqlalchemy import orm 
import services as sv


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


