import sqlalchemy as sql
from sqlalchemy import orm
from sqlalchemy.orm import DeclarativeBase

# from sqlalchemy.orm import DeclarativeBase as Base



DB_URL = "sqlite:///./mydatabase.db"

engine = sql.create_engine(DB_URL, connect_args={"check_same_thread": False})

sessionLocal = orm.sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass
