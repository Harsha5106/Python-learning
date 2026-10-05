from fastapi import FastAPI
from MOCKDATA import products
from dtos import product_dto

app = FastAPI()

@app.get("/")# get router 
def Home():
    return "AVENGERS ASSEMBLE"

@app.get("/about")# about router 
def about():
    return "We are the saviours of the world"

@app.get("/products")
def get_products():
    return products

# path params 
@app.get("/products/{product_id}")
def get_one_product(product_id:int):

   for oneproduct in products:
      if oneproduct.get("id") == product_id:
       return oneproduct

# query parameter 
@app.get("/greet")
def greet_user(name:str):
    return{
         "greet": f"Hello {name}, How are u"
    }
    
# post method 
@app.post("/create_product")
def create_product(data:product_dto):
    data = data.model_dump()
    products.append(data)

    return {"status":"product created sucessfully"}