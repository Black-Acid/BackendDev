from sqlalchemy import (Column, Integer, ForeignKey, String, Float)
from services import Base
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserModel(Base):
    __tablename__ = "Users"
    id = Column(Integer, primary_key=True)
    username = Column(String(255))
    hashed_password = Column(String(255))
    balance = Column(Float, default=0.0)
    
    
    def password_verification(self, password):
        return pwd_context.verify(password, self.hashed_password)