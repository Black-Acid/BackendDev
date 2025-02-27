from sqlalchemy.orm import DeclarativeBase
from database import engine

class Base(DeclarativeBase):
    pass


def create_db():
    Base.metadata.create_all(engine)