import json

student = {
    "name": "Harsha",
    "age": 21,
    "marks": 85
}
'''
json_data = json.dumps(student)

print(json_data)

python_data = json.loads(json_data)
print(python_data)
print(python_data["name"])
print(python_data["marks"])'''

with open("student.json", "w") as file:
    json.dump(student, file, indent=4)

with open("student.json", "r") as file:
    student_data = json.load(file)

print(student_data)
