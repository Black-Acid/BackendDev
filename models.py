from sqlalchemy import (Column, Integer, ForeignKey, String, Float, Text, DateTime)
from database import Base
from passlib.context import CryptContext
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserModel(Base):
    __tablename__ = "Users"
    id = Column(Integer, primary_key=True)
    username = Column(String(255))
    hashed_password = Column(String(255))
    balance = Column(Float, default=0.0)
    messages = relationship("UserMessages", back_populates="user")
    
    
    def password_verification(self, password):
        return pwd_context.verify(password, self.hashed_password)
    
    

class UserMessages(Base):
    __tablename__ = "UserMessages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("Users.id"))
    enquiry = Column(Text)
    response = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    user = relationship("UserModel", back_populates="messages")
    
    