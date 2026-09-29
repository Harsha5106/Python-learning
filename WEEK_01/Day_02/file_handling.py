import csv
from pathlib import Path

students = [
    ["ID", "Name", "Course", "Marks"],
    [101, "Harsha", "Biotechnology", 85],
    [102, "Rahul", "Computer Science", 90],
    [103, "Anjali", "Information Science", 78]
]

# Create the CSV in the same folder as this Python file
file_path = Path(__file__).parent / "students.csv"

# Write data
with open(file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(students)

print("Student data written successfully.")

# Read data
print("\nStudent Records:")

with open(file_path, "r") as file:
    reader = csv.reader(file)

    for row in reader:
        print(row)