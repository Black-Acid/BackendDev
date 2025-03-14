from database import engine, sessionLocal, Base

from sqlalchemy import orm
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, security, Depends
from passlib.context import CryptContext
from groq import Groq

import schemas as sma
import models
import jwt
import pika
import os



client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

RABBITMQ_HOST = "localhost"
QUEUE_NAME = "user_messages"


# Establish a connection to rabbitMq



# Declare a queue



pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET = "nsand329324nrlksndlak;asjdoiqw2"
oauth2schema = security.OAuth2PasswordBearer("/api/login")

def create_db():
    Base.metadata.create_all(engine)
    


def get_db():
    db = sessionLocal()
    
    try:
        yield db
    finally:
        db.close()
        
        
async def get_user(user: sma.UserRequest, db: orm.Session):
    return db.query(models.UserModel).filter_by(username=user.username).first()


async def create_user(user: sma.UserRequest, db: orm.Session):
    hash_password = pwd_context.hash(user.password)
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
        raise HTTPException(status_code=500, detail=f"Couldn't save the data due to {e._message}")
        
    return new_user


async def create_token(user: models.UserModel):
    user_schema = sma.UserResponse.model_validate(user)
    user_dict = user_schema.model_dump()
    
    # if the token is created with the useruser model then I think it will be wise 
    # to remove the balance since the balance can change frequently
    
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
        user = db.query(models.UserModel).get(payload["id"])
    except SQLAlchemyError as e:
        raise HTTPException(status_code=401, detail=f"Invalid credentials{e._message}")
    
    return sma.UserResponse.model_validate(user)
    
    

async def save_queries(
    conversation: sma.UserQueryResponse, 
    db: orm.Session,
    current_user : int
):
    try: 
        collected_data = models.UserMessages(
            user_id = current_user,
            enquiry = conversation.message,
            response = conversation.response
        )
        db.add(collected_data)
        db.commit()
        db.refresh(collected_data)
    except SQLAlchemyError as exception:
        db.rollback()
        raise HTTPException(status_code=401, detail=f"Data could not saved due to {exception._message}")


def get_response(message):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message
            }
        ],
        model = "llama-3.3-70b-versatile",
        stop=None,
        stream=False
    )
    
    return chat_completion.choices[0].message.content


def get_rabbitmq_connection():
    return pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))

def push_to_rabbitmq(data: sma.RabbitMQPush):
    connection = get_rabbitmq_connection()
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME)
    final_message = f"{data.id}|{data.message}"
    
    # Push the data
    channel.basic_publish(exchange="", routing_key=QUEUE_NAME, body=final_message)
    connection.close()
    
    
    
    
 