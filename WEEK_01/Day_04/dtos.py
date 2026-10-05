from pydantic import BaseModel

class product_dto(BaseModel):
    id: int
    title :str
    name :str
    prize : int = 0
    Location : str