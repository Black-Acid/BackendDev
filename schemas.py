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