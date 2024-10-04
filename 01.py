import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_blobs
def scatter_plot(X,y):
    colors = ['red', 'blue', 'green']
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], c=colors[y[i]])


X, y = make_blobs(n_samples=200, n_features=2, cluster_std=2.0)
df = pd.DataFrame(X)
df.columns = ["Width", "Height"]
df["Quarter"] = 0
df["Quarter"][(df["Height"] > 0) * (df["Width"]>0)] = 1
# df["Quarter"][(df["Height"] > 0) * (df["Width"]>0)] = 2
# df["Quarter"][(df["Height"] < 0) * (df["Width"]<0)] = 3
# df["Quarter"][(df["Height"] > 0) * (df["Width"]>0)] = 4
# df["Right"] = 0
# df[df["Width"]>0]
# df["Up"].where(1)


df["y"] = y

# plt.subplot(2,3,1)
# plt.title("Real Data")
# scatter_plot(X,y)
#
#
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)
#
print("Logistic Regression")
logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)
predict = logistic_model.predict(X)
df["Logistic"] = predict
score = logistic_model.score(x_test, y_test) * 100
print(score)
# plt.subplot(2,3,2)
# plt.title(f"Logistic Regression - Score = {score}")
# scatter_plot(X,y)
#
print(f"   Real Data : STD = {np.std(y)}, Mean = {np.mean(y)}")
print(f"Predict Data : STD = {np.std(predict)}, Mean = {np.mean(predict)}")



print("Knn")
knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)
predict = knn_model.predict(X)
df["KNN"] = predict
score = knn_model.score(x_test, y_test) * 100
print(score)
# plt.subplot(2,3,3)
# plt.title(f"Knn - Score = {score}")
# scatter_plot(X,y)
#
print("Svm")
svm_model = SVC()
svm_model.fit(x_train, y_train)
predict = svm_model.predict(X)
df["SVM"] = predict
score = svm_model.score(x_test, y_test) * 100
print(score)
# plt.subplot(2,3,5)
# plt.title(f"Svm - Score = {score}")
# scatter_plot(X,y)
#
print("MLP")
mlp_model = MLPClassifier()
mlp_model.fit(x_train, y_train)
predict = mlp_model.predict(X)
df["MLP"] = predict
score = mlp_model.score(x_test, y_test) * 100
print(score)
# plt.subplot(2,3,6)
# plt.title(f"MLP - Score = {score}")
# scatter_plot(X,y)



print(df)
# df.to_csv("Classification-Comparation .csv")
# print(df.corr())

# plt.show()