from database import engine, sessionLocal, Base
import schemas as sma
from sqlalchemy import orm
import models
import passlib.hash as hash




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
    new_user = models.UserModel(
        username=user.username,
        hashed_password=hash_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user