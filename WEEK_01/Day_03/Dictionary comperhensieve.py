students = {
    "Harsha": 85,
    "Rahul": 90,
    "Anil": 78,
    "Kiran": 65
}

passed_students = {
    name: marks
    for name, marks in students.items()
    if marks >= 70
}

print("Passed students:", passed_students)