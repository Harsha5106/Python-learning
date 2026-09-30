import requests

url = "https://jsonplaceholder.typicode.com/users"

response = requests.get(url)

print("Status code:", response.status_code)

data = response.json()

names = [user["name"] for user in data]

print("Names:", names)