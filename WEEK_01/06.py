Marks  = int(input("Enter the Marks of the student : "))

if( Marks >= 90 and Marks <= 100):
    print("The Student got  A grade")
elif( Marks <= 89 and Marks >= 80):
    print("The Student got  B grade")
elif( Marks <= 79 and Marks >=70):
    print("The Student got  C grade")
elif( Marks <= 69 and Marks >= 60):
    print("The Student got  D grade")
elif( Marks < 60 ):
    print("Unfortunately , the student hs failed ")