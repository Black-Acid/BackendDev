import fastapi
import schemas as sma
from sqlalchemy import orm 
import services as sv


app = fastapi.FastAPI()

@app.post("/api/register-user/")
async def register_user(user: sma.UserRequest, db: orm.Session = fastapi.Depends(sv.get_db) ):
    checked_user = await sv.get_user(user, db)
    if checked_user:
        raise fastapi.HTTPException(status_code=400, detail="A User already exists with that username")
    creating_user = await sv.create_user(user, db)
    
    return creating_user
