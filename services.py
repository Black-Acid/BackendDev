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
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI






# Load FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = FAISS.load_local(
    "theBook_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


print("Retriever type:", type(retriever))




# Implementing the logice for the conversations

from typing import Dict, List
import re

user_sessions: Dict[str, Dict[str, List[str]]] = {}
import re

user_sessions = {}  # Global session tracker


from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re

user_sessions = {}

# async def handle_conversation(user_id: str, message: str, retriever):
#     session = user_sessions.get(user_id)

#     # If this is a new conversation
#     if session is None:
#         # Fetch relevant docs using retriever
#         docs = retriever.invoke(message)

#         # Combine retrieved text
#         context = "\n\n".join([
#             d.page_content for d in docs
#             if hasattr(d, "page_content") and isinstance(d.page_content, str)
#         ])

#         # Prompt template
#         prompt_template = """You are a pharmacy assistant AI.
#             Your job is to ask follow-up questions to the patient before recommending medication.

#             Use the following section of the pharmacology guide as your reference.
#             Ask only relevant diagnostic or clarification questions — do not provide any answers or drug names yet.

#             Context from book:
#             {context}

#             Patient symptom:
#             {symptom}

#             Generate all the relevant diagnostic and clarification questions you would need to accurately understand the patient's condition before recommending treatment:
#             """

#         # Prepare final prompt
#         prompt = prompt_template.format(context=context, symptom=message)

#         # Call ChatGPT 4o-mini directly (async)
#         llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
#         response = await llm.ainvoke(prompt)

#         # Extract text
#         questions_text = response.content.strip()

#         # Split questions line by line
#         lines = [line.strip() for line in questions_text.split("\n") if len(line.strip()) > 8]
#         questions = [q for q in lines if q.endswith("?")]

#         if not questions:
#             return "Sorry, I couldn’t find relevant diagnostic questions for that symptom."

#         # Save conversation state
#         user_sessions[user_id] = {
#             "current_question": 0,
#             "questions": questions,
#             "answers": []
#         }

#         # Return first question
#         return f"Okay, let's start with a few questions.\n{questions[0]}"

#     # If user is answering
#     else:
#         session["answers"].append(message)
#         session["current_question"] += 1

#         if session["current_question"] < len(session["questions"]):
#             next_q = session["questions"][session["current_question"]]
#             return next_q

#         # All questions answered
#         summary = "\n".join(
#             f"Q{i+1}: {q}\nA: {a}"
#             for i, (q, a) in enumerate(zip(session["questions"], session["answers"]))
#         )

#         del user_sessions[user_id]

#         return (
#             f"Thanks for your responses. Here's a summary:\n\n{summary}\n\n"
#             "I’ll now analyze this and forward it to the pharmacist for a suitable recommendation."
#         )
from typing import Optional

# Optional: if you want LLM fallback, set USE_LLM_FALLBACK = True and provide `llm_call` function.
USE_LLM_FALLBACK = False

# keywords that strongly indicate non-answer intents
REQUEST_HELP_KEYWORDS = [
    "pharmacist", "connect me", "connect to", "prescription", "get medicine",
    "i want medicine", "i need medicine", "talk to a pharmacist", "talk to a doctor",
    "send to pharmacist", "call a pharmacist"
]
CHANGE_TOPIC_KEYWORDS = ["stop", "enough", "i'm tired", "im tired", "i am tired", "pause", "quit", "change topic", "new topic"]
NEW_SYMPTOM_KEYWORDS = ["also", "another", "new", "besides", "i also have", "i have also"]

YES_NO_SHORT = {"yes","no","y","n","yeah","yep","nah","nope","true","false"}

def _contains_any(text: str, keywords):
    t = text.lower()
    return any(k in t for k in keywords)


# improved detect_intent
# ---------- Improved detect_intent (replacement) ----------
SYMPTOM_TRIGGERS = [
    "i have", "i'm having", "i am having", "i am", "i'm", "i've been", "i have been",
    "experiencing", "suffering from", "been having"
]

COMMON_SYMPTOM_WORDS = [
    "headache", "cough", "fever", "pain", "nausea", "vomit", "vomiting", "dizzy",
    "dizziness", "chest pain", "sore throat", "rash", "diarrhea", "shortness of breath",
    "breath", "bleeding", "swelling"
]

QUESTION_WORDS = {"what", "when", "where", "why", "who", "how", "which", "whom"}

def _contains_any_phrase(text: str, phrases):
    t = text.lower()
    return any(p in t for p in phrases)

def detect_intent(message: str, last_assistant_question: Optional[str] = None, history: str = "") -> str:
    """
    Returns one of:
    - "answer"
    - "new_symptom"
    - "request_help"
    - "change_topic"
    - "chat"
    """
    msg = (message or "").strip()
    if not msg:
        return "chat"

    lower = msg.lower()
    tokens = lower.split()

    # 0) If user asks something about the bot or uses "you", treat as chat/change_topic
    if any(word in lower for word in ["your name", "who are you", "what are you", "you?" ,"you are"]) or "your" in tokens[:3]:
        return "chat"

    # NEW: Detect explicit symptom patterns early
    if _contains_any_phrase(lower, SYMPTOM_TRIGGERS) and not _contains_any(lower, CHANGE_TOPIC_KEYWORDS):
        return "new_symptom"
    if _contains_any_phrase(lower, COMMON_SYMPTOM_WORDS) and len(tokens) > 1:
        return "new_symptom"

    # Strong keyword checks (non-answer)
    if _contains_any(lower, REQUEST_HELP_KEYWORDS):
        return "request_help"
    if _contains_any(lower, CHANGE_TOPIC_KEYWORDS):
        return "change_topic"
    if _contains_any(lower, NEW_SYMPTOM_KEYWORDS):
        return "new_symptom"

    # If the user directly asks a question (ends with ? or starts with question word)
    is_question_mark = msg.endswith("?")
    starts_with_qword = tokens and tokens[0] in QUESTION_WORDS

    if is_question_mark or starts_with_qword:
        # If the question *exactly* repeats the assistant's last question -> it's likely an answer rephrase
        if last_assistant_question and msg.lower().strip() == last_assistant_question.strip().lower():
            return "answer"
        # Otherwise treat as chat (user changed topic or asked about bot or clarified)
        return "chat"

    # Very short single-word or short phrases -> likely quick answers (yes/no, short counts)
    if len(tokens) <= 4:
        token0 = tokens[0].strip(".,!?")
        if token0 in YES_NO_SHORT:
            return "answer"
        if re.match(r"^\d+$", token0) or re.match(r"^\d+(?:\.\d+)?(days|day|hrs|hours|weeks)?$", lower) or re.match(r"^(one|two|three|four|five|six|seven|eight|nine|ten)$", token0):
            return "answer"
        if _contains_any(lower, CHANGE_TOPIC_KEYWORDS):
            return "change_topic"
        # short messages that clearly reference symptoms like "headache" alone -> treat as new_symptom
        if tokens[0] in COMMON_SYMPTOM_WORDS:
            return "new_symptom"

    # If last assistant question exists, check semantic fit: does the message answer that question (heuristic)
    if last_assistant_question:
        la = last_assistant_question.lower()
        # duration question -> numeric answer
        if re.search(r"(how long|when did|duration|since when)", la) and re.search(r"\b(days?|hours?|weeks?|months?)\b", lower):
            return "answer"
        # yes/no style question -> short message -> answer
        if re.match(r'^(do|did|is|are|have|has|can|could|should|was|were)\b', la.strip()):
            if len(tokens) <= 6:
                return "answer"
        # if last question asks about location/time and user replies with a full sentence starting with 'i' -> answer
        if re.search(r"(where|when|which)", la) and tokens and tokens[0] in {"i","i've","i'm","i am"}:
            return "answer"

    # Heuristic: long messages -> chat/new_symptom
    if len(tokens) > 8:
        if _contains_any(lower, ["i also have", "also have", "and also", "also experiencing", "another symptom", "besides that"]):
            return "new_symptom"
        return "chat"

    # FINAL fallback: be more conservative — prefer chat unless there's a clear last question context
    if last_assistant_question and len(tokens) <= 6:
        # short replies when assistant asked something recently -> probably an answer
        return "answer"

    return "chat"


# def detect_intent(message: str, last_assistant_question: Optional[str] = None, history: str = "") -> str:
#     """
#     Returns one of:
#     - "answer"
#     - "new_symptom"
#     - "request_help"
#     - "change_topic"
#     - "chat"
#     """
#     msg = message.strip()
#     if not msg:
#         return "chat"

#     lower = msg.lower()

#     # 1) Strong keyword checks (non-answer)
#     if _contains_any(lower, REQUEST_HELP_KEYWORDS):
#         return "request_help"
#     if _contains_any(lower, CHANGE_TOPIC_KEYWORDS):
#         return "change_topic"
#     if _contains_any(lower, NEW_SYMPTOM_KEYWORDS):
#         return "new_symptom"

#     # 2) If the user directly asks a question (likely they changed topic)
#     if msg.endswith("?"):
#         # If the question repeats the assistant's last question, treat as answer attempt
#         if last_assistant_question and msg.lower() == last_assistant_question.strip().lower():
#             return "answer"
#         return "chat"

#     # 3) Very short single-word or short phrases -> likely quick answers (yes/no, short counts)
#     tokens = msg.split()
#     if len(tokens) <= 4:
#         # If short token is clearly a yes/no or numeric answer, treat as answer
#         token0 = tokens[0].lower().strip(".,!?")
#         if token0 in YES_NO_SHORT:
#             return "answer"
#         # numeric responses (e.g., "2 days", "3", "two")
#         if re.match(r"^\d+$", token0) or re.match(r"^\d+(?:\.\d+)?(days|day|hrs|hours|weeks)?$", lower) or re.match(r"^(one|two|three|four|five|six|seven|eight|nine|ten)$", token0):
#             return "answer"
#         # short "I’m tired" type: check change-topic list
#         if _contains_any(lower, CHANGE_TOPIC_KEYWORDS):
#             return "change_topic"

#     # 4) If last assistant question exists, check semantic fit: does the message answer that question (heuristic)
#     if last_assistant_question:
#         la = last_assistant_question.lower()
#         # if last question asked for duration and user says X days/hours -> answer
#         if re.search(r"(how long|when did|duration|since when)", la) and re.search(r"\b(days?|hours?|weeks?|months?)\b", lower):
#             return "answer"
#         # if last question asked yes/no style (starts with "do", "did", "is", "are", "have") and message short -> answer
#         if re.match(r'^(do|did|is|are|have|has|can|could|should|was|were)\b', la.strip()):
#             if len(tokens) <= 6:
#                 return "answer"

#     # 5) Heuristic fallback: if message contains many words and a sentence, treat as chat/new_symptom
#     if len(tokens) > 8:
#         # If it reads like adding another symptom: "I also have chest pain and shortness of breath"
#         if _contains_any(lower, ["i also have", "also have", "and also", "also experiencing", "another symptom", "besides that"]):
#             return "new_symptom"
#         return "chat"

#     # 6) Final fallback: treat as answer (conservative)
#     return "answer"


# Optional LLM fallback (not enabled by default)
async def detect_intent_with_llm(message: str, history: str, llm_client) -> str:
    """
    If you want to call an LLM for ambiguous cases, implement this with your llm_client.
    llm_client should provide .ainvoke(prompt) -> response.content
    """
    prompt = f"""
Classify the user's intent into one of: answer, new_symptom, request_help, change_topic, chat.
Conversation history:
{history}

User message:
{message}

Return a single word.
"""
    r = await llm_client.ainvoke(prompt)
    return r.content.strip().lower()


async def handle_intent_with_ai(intent: str, message: str, history: str, session) -> str:
    """
    Generates dynamic responses for detected intents using a small AI model.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    prompt_text = """
    You are a friendly, intelligent pharmacy assistant chatbot.
    You already know the user's intent: {intent}.
    Here is the recent conversation history:
    {history}

    User message:
    {message}

    Your job:
    - If intent = "chat": respond casually but gently guide them back to describing their symptoms.
    - If intent = "request_help": acknowledge their request and assure them you'll connect them to a pharmacist.
    - If intent = "change_topic": ask what new topic or issue they’d like to discuss.
    - If intent = "new_symptom": acknowledge it and ask one initial clarifying question.
    - If intent = "answer": confirm their response and ask the next diagnostic question if available.
    Be natural and human-like, not robotic.
    Respond in 2–3 sentences only.
    """

    prompt = PromptTemplate.from_template(prompt_text)
    result = await llm.ainvoke(prompt.format(intent=intent, message=message, history=history))
    return result.content.strip()

user_sessions = {}

async def handle_conversation(user_id: str, message: str, retriever):
    session = user_sessions.get(user_id)

    # Build conversation history for context
    history = ""
    if session:
        history = "\n".join(
            f"Q{i+1}: {q}\nA: {a}"
            for i, (q, a) in enumerate(zip(session["questions"], session["answers"]))
        )

    # Get the last assistant question (if any)
    last_question = None
    if session and session["current_question"] < len(session["questions"]):
        last_question = session["questions"][session["current_question"]]

    # --- Detect intent ---
    # ✅ Fix: if no session exists, assume the first message is a symptom
    if session is None:
        intent = "new_symptom"
    else:
        intent = detect_intent(message, last_question, history)

    print(f"[Intent detected: {intent}]")

    # --- CASE 1: NEW CONVERSATION OR NEW SYMPTOM ---
    if session is None or intent == "new_symptom":
        docs = retriever.invoke(message)
        context = "\n\n".join([
            d.page_content for d in docs
            if hasattr(d, "page_content") and isinstance(d.page_content, str)
        ])

        prompt_template = """You are a pharmacy assistant AI.
        Your job is to ask follow-up questions to the patient before recommending medication.

        Use the following section of the pharmacology guide as your reference.
        Ask only relevant diagnostic or clarification questions — do not provide any answers or drug names yet.

        Context from book:
        {context}

        Patient symptom:
        {symptom}

        Generate all the relevant diagnostic and clarification questions you would need to accurately understand the patient's condition before recommending treatment:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "symptom"])
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

        result = await llm.ainvoke(prompt.format(context=context, symptom=message))
        questions_text = result.content if hasattr(result, "content") else str(result)

        lines = [line.strip() for line in questions_text.split("\n") if len(line.strip()) > 8]
        questions = [q for q in lines if q.endswith("?")]

        if not questions:
            return "Sorry, I couldn’t find relevant diagnostic questions for that symptom."

        user_sessions[user_id] = {
            "current_question": 0,
            "questions": questions,
            "answers": []
        }

        return f"Okay, let's start with a few questions.\n{questions[0]}"

    # --- CASE 2: USER IS ANSWERING ---
    elif intent == "answer" and session:
        session["answers"].append(message)
        session["current_question"] += 1

        if session["current_question"] < len(session["questions"]):
            next_q = session["questions"][session["current_question"]]
            return next_q
        else:
            summary = "\n".join(
                f"Q{i+1}: {q}\nA: {a}"
                for i, (q, a) in enumerate(zip(session["questions"], session["answers"]))
            )
            del user_sessions[user_id]
            return (
                f"Thanks for your responses. Here's a summary:\n\n{summary}\n\n"
                "I’ll now analyze this and forward it to the pharmacist for a suitable recommendation."
            )

    # --- CASES 3, 4, 5 (AI handles these dynamically) ---
    else:
        # Whether request_help, change_topic, or casual chat — let the AI reply contextually
        ai_reply = await handle_intent_with_ai(intent, message, history, session)
        if session and session["current_question"] < len(session["questions"]):
            next_q = session["questions"][session["current_question"]]
            return f"{ai_reply}\n\nAnyway, let;s continue:\n{next_q}"
        else:
            return ai_reply

# async def handle_conversation(user_id: str, message: str, retriever):
#     session = user_sessions.get(user_id)

#     # Build conversation history for context
#     history = ""
#     if session:
#         history = "\n".join(
#             f"Q{i+1}: {q}\nA: {a}"
#             for i, (q, a) in enumerate(zip(session["questions"], session["answers"]))
#         )

#     # Get the last assistant question (if any)
#     last_question = None
#     if session and session["current_question"] < len(session["questions"]):
#         last_question = session["questions"][session["current_question"]]

#     # --- Detect intent ---
#     intent = detect_intent(message, last_question, history)
#     print(f"[Intent detected: {intent}]")

#     # --- CASE 1: NEW CONVERSATION OR NEW SYMPTOM ---
#     if session is None or intent == "new_symptom":
#         docs = retriever.invoke(message)
#         context = "\n\n".join([
#             d.page_content for d in docs
#             if hasattr(d, "page_content") and isinstance(d.page_content, str)
#         ])

#         prompt_template = """You are a pharmacy assistant AI.
#         Your job is to ask follow-up questions to the patient before recommending medication.

#         Use the following section of the pharmacology guide as your reference.
#         Ask only relevant diagnostic or clarification questions — do not provide any answers or drug names yet.

#         Context from book:
#         {context}

#         Patient symptom:
#         {symptom}

#         Generate all the relevant diagnostic and clarification questions you would need to accurately understand the patient's condition before recommending treatment:
#         """

#         prompt = PromptTemplate(template=prompt_template, input_variables=["context", "symptom"])
#         llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

#         result = await llm.ainvoke(prompt.format(context=context, symptom=message))
#         questions_text = result.content if hasattr(result, "content") else str(result)

#         lines = [line.strip() for line in questions_text.split("\n") if len(line.strip()) > 8]
#         questions = [q for q in lines if q.endswith("?")]

#         if not questions:
#             return "Sorry, I couldn’t find relevant diagnostic questions for that symptom."

#         user_sessions[user_id] = {
#             "current_question": 0,
#             "questions": questions,
#             "answers": []
#         }

#         return f"Okay, let's start with a few questions.\n{questions[0]}"

#     # --- CASE 2: USER IS ANSWERING ---
#     elif intent == "answer" and session:
#         session["answers"].append(message)
#         session["current_question"] += 1

#         if session["current_question"] < len(session["questions"]):
#             next_q = session["questions"][session["current_question"]]
#             return next_q
#         else:
#             summary = "\n".join(
#                 f"Q{i+1}: {q}\nA: {a}"
#                 for i, (q, a) in enumerate(zip(session["questions"], session["answers"]))
#             )
#             del user_sessions[user_id]
#             return (
#                 f"Thanks for your responses. Here's a summary:\n\n{summary}\n\n"
#                 "I’ll now analyze this and forward it to the pharmacist for a suitable recommendation."
#             )

#     # --- CASE 3: USER REQUESTS PHARMACIST ---
#     elif intent == "request_help":
#         if user_id in user_sessions:
#             del user_sessions[user_id]
#         return "Okay, I’ll connect you to a pharmacist now for further assistance."

#     # --- CASE 4: USER CHANGES TOPIC ---
#     elif intent == "change_topic":
#         if user_id in user_sessions:
#             del user_sessions[user_id]
#         return "Sure, we can stop here. What would you like to discuss instead?"

#     # --- CASE 5: CASUAL CHAT ---
#     else:
#         return "Haha, I get you! But let's stay focused for a bit — could you tell me more about your symptoms?"


# async def handle_conversation(user_id: str, message: str, retriever):
#     session = user_sessions.get(user_id)

#     # Build conversation history for context
#     history = ""
#     if session:
#         history = "\n".join(
#             f"Q{i+1}: {q}\nA: {a}"
#             for i, (q, a) in enumerate(zip(session["questions"], session["answers"]))
#         )

#     # 🔹 Detect user intent first
#     intent = detect_intent(message, history)
#     print(f"[Intent detected: {intent}]")

#     # --- CASE 1: NEW CONVERSATION OR NEW SYMPTOM ---
#     if session is None or intent == "new_symptom":
#         docs = retriever.invoke(message)
#         context = "\n\n".join([
#             d.page_content for d in docs
#             if hasattr(d, "page_content") and isinstance(d.page_content, str)
#         ])

#         prompt_template = """You are a pharmacy assistant AI.
#         Your job is to ask follow-up questions to the patient before recommending medication.

#         Use the following section of the pharmacology guide as your reference.
#         Ask only relevant diagnostic or clarification questions — do not provide any answers or drug names yet.

#         Context from book:
#         {context}

#         Patient symptom:
#         {symptom}

#         Generate all the relevant diagnostic and clarification questions you would need to accurately understand the patient's condition before recommending treatment:
#         """

#         prompt = PromptTemplate(template=prompt_template, input_variables=["context", "symptom"])
#         llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

#         result = await llm.ainvoke(prompt.format(context=context, symptom=message))
#         questions_text = result.content if hasattr(result, "content") else str(result)

#         lines = [line.strip() for line in questions_text.split("\n") if len(line.strip()) > 8]
#         questions = [q for q in lines if q.endswith("?")]

#         if not questions:
#             return "Sorry, I couldn’t find relevant diagnostic questions for that symptom."

#         user_sessions[user_id] = {
#             "current_question": 0,
#             "questions": questions,
#             "answers": []
#         }

#         return f"Okay, let's start with a few questions.\n{questions[0]}"

#     # --- CASE 2: USER IS ANSWERING ---
#     elif intent == "answer" and session:
#         session["answers"].append(message)
#         session["current_question"] += 1

#         if session["current_question"] < len(session["questions"]):
#             return session["questions"][session["current_question"]]
#         else:
#             summary = "\n".join(
#                 f"Q{i+1}: {q}\nA: {a}"
#                 for i, (q, a) in enumerate(zip(session["questions"], session["answers"]))
#             )
#             del user_sessions[user_id]
#             return (
#                 f"Thanks for your responses. Here's a summary:\n\n{summary}\n\n"
#                 "I’ll now analyze this and forward it to the pharmacist for a suitable recommendation."
#             )

#     # --- CASE 3: USER REQUESTS PHARMACIST ---
#     elif intent == "request_help":
#         if user_id in user_sessions:
#             del user_sessions[user_id]
#         return "Okay, I’ll connect you to a pharmacist now for further assistance."

#     # --- CASE 4: USER CHANGES TOPIC ---
#     elif intent == "change_topic":
#         if user_id in user_sessions:
#             del user_sessions[user_id]
#         return "Sure, we can stop here. What would you like to discuss instead?"

#     # --- CASE 5: CASUAL CHAT ---
#     else:
#         return "Haha, I get you! But let's stay focused for a bit — could you tell me more about your symptoms?"






# Template for generating patient questions
prompt_template = """You are a pharmacy assistant AI.
    Your job is to ask follow-up questions to the patient before recommending medication.

    Use the following section of the pharmacology guide as your reference.
    Ask only relevant diagnostic or clarification questions — do not provide any answers or drug names yet.

    Context from book:
    {context}

    Patient symptom:
    {symptom}

    Generate all the relevant diagnostic and clarification questions you would need to accurately understand the patient's condition before recommending treatment:
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
    
    
from transformers import pipeline


intent_model = pipeline("text2text-generation", model="google/flan-t5-small")

