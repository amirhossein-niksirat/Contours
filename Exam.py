import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import f1_score
from sklearn.datasets import make_blobs



f = np.load('dataset.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']

x_train = np.array([np.ravel(x) for x in x_train])
x_test = np.array([np.ravel(x) for x in x_test])

df = pd.DataFrame(x_train)
df_1 = pd.DataFrame(y_train)
print(df)
print(df_1.max())



def scatter_plot(X,y):
    colors = ['red', 'blue', 'green']
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1],c=colors[y[i]])

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

        print("GB-Classifier")
        GB_model = GradientBoostingClassifier()
        GB_model.fit(x_train, y_train)
        predict = GB_model.predict(X)
        df["GB-clasiifier"] = predict
        score = GB_model.score(x_test, y_test) * 100
        print(score)

        print(df)
        # df.to_csv("Classification-Comparation .csv")
        print(df.corr())

        plt.show()




