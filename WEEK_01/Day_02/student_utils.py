def calculate_average(marks):
    return sum(marks) / len(marks)


def check_result(marks):
    average = calculate_average(marks)

    if average >= 40:
        return "Pass"
    else:
        return "Fail"