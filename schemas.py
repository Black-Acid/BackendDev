import pydantic

class UserRequest(pydantic.BaseModel):
    username: str
    password: str
    
class UserResponse(UserRequest):
    id : int
    balance: int 