import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.decomposition import PCA


df=pd.read_csv("students.csv")
print("Dataset Preview:")
print(df.head())

x=df[['Hours_Study','Attendance','Assignments','Marks']]
y=df[['Result']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=False)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
predict=knn.predict(x_test)
print(accuracy_score(y_test,predict))

svm = SVC(kernel='linear')
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))


per = Perceptron()
per.fit(x_train, y_train)
per_pred = per.predict(x_test)
print("Perceptron Accuracy:", accuracy_score(y_test, per_pred))

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(x)

print("\nClustered Data:")
print(df[["Hours_Study","Attendance","Cluster"]])
center=kmeans.cluster_centers_
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x)
center_pca = pca.transform(center)
plt.scatter(X_pca[:, 0], X_pca[:, 1],c=df["Cluster"], cmap='viridis')
plt.scatter(center_pca[:, 0], center_pca[:, 1], s=200, c='red', marker='X')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering (PCA Visualization)")
plt.show()

states = 3   
actions = 3  

Q = np.zeros((states, actions))

alpha = 0.1
gamma = 0.9

for episode in range(100):

    state = np.random.randint(0, 3)

    for step in range(5):

        action = np.random.randint(0, 3)

        
        next_state = min(2, max(0, state + action - 1))

        reward = 1 if next_state == 2 else 0

        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

print("\nQ-Learning Table (Study Strategy):")
print(Q)


new_student = [[5,80,7,70]]

print("\nPredictions for new student:")
print("KNN:", knn.predict(new_student))
print("SVM:", svm.predict(new_student))
print("Perceptron:", per.predict(new_student))