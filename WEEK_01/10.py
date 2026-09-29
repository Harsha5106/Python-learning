# Mini Student Report
# Concepts: Variables + lists + conditions + loops + functions


def calculate_grade(avg):
    if avg >= 90:
        return 'A'
    elif avg >= 80:
        return 'B'
    elif avg >= 70:
        return 'C'
    elif avg >= 60:
        return 'D'
    else:
        return 'F'


def check_result(avg):
    if avg >= 40:
        return 'PASS'
    else:
        return 'FAIL'


student_name = input("Enter student name: ")
student_age = int(input("Enter student age: "))

subjects = ["Python", "SQL", "Maths", "Physics", "Biology"]
marks = []

for subject in subjects:
    mark = int(input(f"Enter marks for {subject}: "))
    marks.append(mark)

total = sum(marks)
average = total / len(marks)
grade = calculate_grade(average)
result = check_result(average)

print("\n========== STUDENT REPORT ==========")
print(f"\nName: {student_name}")
print(f"Age: {student_age}\n")
print("Marks:")

for subject, mark in zip(subjects, marks):
    print(f"{subject}: {mark}")

print(f"\nTotal: {total}")
print(f"Average: {average:.1f}")
print(f"Grade: {grade}")
print(f"Result: {result}")
