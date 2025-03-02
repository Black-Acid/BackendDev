from database import engine, sessionLocal, Base
import schemas as sma
from sqlalchemy import orm
import models
import passlib.hash as hash
import jwt
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, security, Depends

JWT_SECRET = "nsand329324nrlksndlak;asjdoiqw2"
oauth2schema = security.OAuth2PasswordBearer("/api/login")

def create_db():
    Base.metadata.create_all(engine)
    
create_db()
def get_db():
    db = sessionLocal()
    
    try:
        yield db
    finally:
        db.close()
        
        
async def get_user(user: sma.UserRequest, db: orm.Session):
    return db.query(models.UserModel).filter_by(username=user.username).first()


async def create_user(user: sma.UserRequest, db: orm.Session):
    hash_password = hash.bcrypt.hash(user.password)
    try:
        
        new_user = models.UserModel(
            username=user.username,
            hashed_password=hash_password
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Couldn't save the data due to {e}")
        
    return new_user


async def create_token(user: models.UserModel):
    user_schema = sma.UserResponse.model_validate(user)
    user_dict = user_schema.model_dump()
    
    token = jwt.encode(user_dict, JWT_SECRET)
    
    return dict(access_token=token, token_type="bearer")

async def login(identifier: str, password: str, db: orm.Session):
    user = db.query(models.UserModel).filter_by(username=identifier).first()
    
    if not user:
        return False
    
    
    if not user.password_verification(password):
        return False
    
    return user


async def get_current_user(db: orm.Session = Depends(get_db), token: str = Depends(oauth2schema)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user = db.query(models.UserModel).get(id=payload["id"])
    except:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return sma.UserResponse.model_validate(user)
    