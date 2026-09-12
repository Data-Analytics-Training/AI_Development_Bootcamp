"""
Machine Learning (ML)

Supervised Learning

Model Input Feature (X)
Corresponding Ground Truth Labels (y)


Core Architecture:
	X (Feature) -> Model -> y (Label)

For example:
	- is the e-mail spam or not?
	- What will the house price be?
	- Should we offer credit for white goods to this customer?

There is a LABEL, which represents the type of learning.
"""

'''
Let's predict whether a student passes the exam based on the folling information:
- Daily student hour
- Class attendance percentage

Lets predict whether they will pass the exam based on this information.

Label:
    0: Fail
    1: Pass 
    
Algorith Used:
    Logistic Regression
    
'''
import numpy as np
from sklearn.linear_model import LogisticRegression

def main():

    X = np.array(
        [
            [1, 30],
            [2, 40],
            [2, 50],
            [3, 55],
            [4, 60],
            [5, 65],
            [6, 75],
            [7, 85],
            [8, 90],
            [9, 95],
        ]
    )

    # y (Labels)
    # The most important feature of supervised learning is presence of Y labels alongside X data
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    print(" SUPERVISED LEARNING (Features(+) Label(+))")
    print("\nX - Students Attributes (Features")
    print(X)

    print("\ny - Labels")
    print(y)

    #! Model Creation
    # LogisticRegression is a classification algorithm
    # There are two classes:
    # 0 -> Failed
    # 1 -> Passed

    # LogisticRegression is a classification algorithm used to predict which of two or more classes an instance belongs to.

    model = LogisticRegression()

    #! Model Training
    # The model sees both the features and the correct answers.
    # In this relationship: Study hours + Attendance rate -> Passed/Failed

    model.fit(X, y)

    new_student = np.array([[6, 80]])

    #! Prediction
    # Example: The student studies for 6 hours, class attendance is 80%
    predict = model.predict(new_student)[0]

    #! Prediction Probability
    probabilities = model.predict_proba(new_student)[0]

    print("\nNew Student")
    print(f"Study Hour: {new_student[0][0]}")
    print(f"Attendance Rate: {new_student[0][1]}")

    #! Conditional
    if predict == 1:
        print("Result: The student is expected to PASS.")
    else:
        print("Result: The student is expected to FAIL.")

    print("\nProbability")
    print(f"Probability of failing: %{probabilities[0] * 100:.2f} ")
    print(f"Probability of passing: %{probabilities[1] * 100:.2f} ")

    # ================ In Summary ==============================
    print("Summary:")
    print("Supervised Learning has LABELS")
    print("DON'T FORGET: The model learns from the correct answers in the past.")
    print("In this example, the labels are 0=Failed, 1=Passed")

if __name__ == '__main__':
    main()
