import pika.exceptions
from database import engine, sessionLocal, Base

from sqlalchemy import orm
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, security, Depends
from passlib.context import CryptContext
from groq import Groq
from openai import OpenAI

import schemas as sma
import models
import jwt
import pika
import os
import asyncio


# packages for the vector database and AI
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings




# loader = PyPDFLoader("theBook.pdf")
# documents = loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# docs = splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# vectorstore = FAISS.from_documents(docs, embeddings)

# vectorstore.save_local("theBook_faiss_index")


# Load FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = FAISS.load_local(
    "theBook_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})



# Template for generating patient questions
prompt_template = """You are a pharmacy assistant AI.
    Your job is to ask follow-up questions to the patient before recommending medication.

    Use the following section of the pharmacology guide as your reference.
    Ask only relevant diagnostic or clarification questions — do not provide any answers or drug names yet.

    Context from book:
    {context}

    Patient symptom:
    {symptom}

    Generate 3 questions you would ask next:
    """

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "symptom"])



client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)
gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


async def get_response(message):
    # actual_message = message.split("|")
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


async def get_response2(message: str):
    
    docs = await retriever.ainvoke(message)  # ✅ async-safe and works with new LangChain
    context = "\n\n".join([d.page_content for d in docs])
    formatted_prompt = prompt.format(context=context, symptom=message)

    
    
    chat_completion = gpt_client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o" if you prefer full GPT-4
        messages=[
            {"role": "system", "content": "You are a pharmacy assistant who asks intelligent, medically correct diagnostic questions based on the provided pharmacology guide."},
            {"role": "user", "content": formatted_prompt}
        ]
    )
    print("fetching response from gpt")
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
    
    
    
async def get_query():
    connection = get_rabbitmq_connection()
    channel = connection.channel()
    
    try:
        queue = channel.queue_declare(queue=QUEUE_NAME, passive=True)
        
        message_count = queue.method.message_count
        
        if message_count:
            tasks = []
            
            
            
            for _ in range(message_count):
                method_frame, header_frame, body = channel.basic_get(queue=QUEUE_NAME)
                
                task = asyncio.create_task(get_response(body.decode(), channel, method_frame.delivery_tag))
                
                tasks.append(task)
            
            return await asyncio.gather(*tasks)
    except pika.exceptions.ChannelClosedByBroker:
        print("Queue cannnot be found")

    finally:
        connection.close()        
    
    
# giving the book to gpt 4o mini as a reference point to study 
