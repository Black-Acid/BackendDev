import pydantic
from pydantic import ConfigDict


class UserBase(pydantic.BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    username: str



class UserRequest(UserBase):
    password: str

    class Config:
        from_attributes = True
    
class UserResponse(UserBase):
    id : int
    balance: int 
    
    class Config:
        from_attributes = True
        

class UserQuery(pydantic.BaseModel):
    message: str
    
    
    
class UserQueryResponse(UserQuery):
    response : str
    
class RabbitMQPush(pydantic.BaseModel):
    id: int
    message: str
    
class RabbitMQPull(RabbitMQPush):
    response: str