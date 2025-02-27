from sqlalchemy.orm import DeclarativeBase
from database import engine, sessionLocal

class Base(DeclarativeBase):
    pass


def create_db():
    Base.metadata.create_all(engine)
    

def get_db():
    db = sessionLocal()
    
    try:
        yield db
    finally:
        db.close()