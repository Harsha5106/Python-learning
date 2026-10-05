from fastapi import FastAPI , HTTPException
from store import todos
from controls import TodoCreate , TodoResponse
import asyncio

app = FastAPI()

@app.get("/todos")
def get_todos(completed: bool = None):

    if completed is None:
        return todos

    return [todo for todo in todos if todo["completed"] == completed] # LIST COMPERHENSIEVE

# used Response model and Post Method 
@app.post("/todos", response_model=TodoResponse)
def create_todo(todo: TodoCreate):
    new_todo = {
        "id": len(todos) + 1,
        "title": todo.title,
        "completed": todo.completed
    }

    todos.append(new_todo)

    return new_todo

# Path Parameter 
@app.get("/todos/{todo_id}")
def todo (todo_id: int):
        for todo in todos:
          if todo["id"] == todo_id:
            return todo
    
        raise HTTPException(status_code=404, detail="Todo not found")

# async  and await 
@app.get("/async-test")
async def async_test():
    await asyncio.sleep(2)
    return {"message": "Async endpoint working"}

# DELETE METHOD 
@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: int):
    for todo in todos:
        if todo["id"] == todo_id:
            todos.remove(todo)
            return {"message": "Todo deleted successfully"}

    raise HTTPException(status_code=404, detail="Todo not found") # EXCEPTION HANDLING 

# UPDATE METHOD 
@app.put("/todos/{todo_id}", response_model=TodoResponse)
def update_todo(todo_id: int, todo: TodoCreate):

    for item in todos:
        if item["id"] == todo_id:
            item["title"] = todo.title
            item["completed"] = todo.completed
            return item

    raise HTTPException(status_code=404, detail="Todo not found") # EXCEPTION HANDLING 