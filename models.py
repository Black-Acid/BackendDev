from sqlalchemy import (Column, Integer, ForeignKey, String, Float)
from services import Base

class UserModel(Base):
    __tablename__ = "Users"
    id = Column(Integer, primary_key=True)
    username = Column(String(255))
    hashed_password = Column(String(255))
    balance = Column(Float, default=0.0)