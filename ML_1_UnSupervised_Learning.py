"""
Machine Learning (ML)

Unsupervised Learning (Feature (+) + Label(-))

Where the model has input data (X) but lacks the corresponding correct labels.



Core Architecture:
X (Features) -> Model -> Groups/Patterns

There is NO LABEL

Model:
- It can group similar records.
- It can find hidden patterns in the data
- It can create customer segments
- It can help detect anomalous data


"""

"""
In this example,
Based on the customers:
- Annual income,
- Monthly spending
We will divide the customers into 3 groups.

However, the model is not given any correct answers, such as 
- this customer belongs to Group A
- this customer belongs to Group B

Algorithm Used:
    KMeans Clustering, StandardScaler
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    # x = Customer features
    # [annual_income, monthly_spending]
    # Note: There is no LABEL here

    X = np.array(
        [
            [20, 10],
            [22, 12],
            [25, 15],
            [50, 45],
            [52, 48],
            [55, 50],
            [85, 80],
            [88, 85],
            [90, 88],
        ]
    )

    print("UNSUPERVISED LEARNING")
    print("\nCustomer Features")
    print(X)

    # ----------------------------------------
    # Scaling
    # StandardScaler, brings numbers of different magnitudes to similar scale.
    # Scaling is beneficial because K-Means performs distance calculations.

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------------------------------------
    # K-Means Model Creation
    model = KMeans(n_clusters=3, random_state=42, n_init=10)

    # Model Training
    # The model both learns and generates a cluster number for the data
    clusters = model.fit_predict(X_scaled)

    print("\nCreated Clusters by Model")

    for i, customer in enumerate(X):
        income = customer[0]
        spending = customer[1]
        cluster = clusters[i]

        print(
            f"Customer {i + 1}: "
            f"Income = {income}, Spending = {spending} "
            f"--> Cluster = {cluster}"
        )


if __name__ == "__main__":
    main()
