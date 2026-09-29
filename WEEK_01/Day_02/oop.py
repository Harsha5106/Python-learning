class Student:
    def __init__(self, name, student_id, marks):
        self.name = name
        self.student_id = student_id
        self.marks = marks

    def display_student(self):
        print(f"Student Name: {self.name}")
        print(f"Student ID: {self.student_id}")
        print(f"Marks: {self.marks}")

class Course:
    def __init__(self, course_name, duration):
        self.course_name = course_name
        self.duration = duration

    def display_course(self):
        print(f"Course: {self.course_name}")
        print(f"Duration: {self.duration}")

class EngineeringStudent(Student):
    def __init__(self, name, student_id, marks, branch):
        super().__init__(name, student_id, marks)
        self.branch = branch

    def display_student(self):
        super().display_student()
        print(f"Branch: {self.branch}")

student = EngineeringStudent(
    "Harsha",
    "BT001",
    85,
    "Biotechnology"
)

course = Course(
    "Python Programming",
    "3 Months"
)

student.display_student()

print()

course.display_course()