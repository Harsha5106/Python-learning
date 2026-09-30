import requests
import json
# imported the requests and json 

# External Data calling / requesting 
url = "https://jsonplaceholder.typicode.com/users"

try:
    response = requests.get(url) #getting response from url 
    response.raise_for_status() # also adding status code 

    data = response.json() #res

    names = [user["name"] for user in data] # list Comperhensieve 

    print("Users:")
    print(names)

    with open("users.json", "w") as file:
       json.dump(names, file, indent=4) # written a json file 

    print("Data saved successfully!") # after creating the file printing data saved 

    print("API request successful!")
    print("Number of users:", len(data))

# Error Handling 
except requests.RequestException as error:
    print("API request failed:", error)