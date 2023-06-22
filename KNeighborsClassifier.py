from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time

X, y_true = make_blobs(n_samples=100,n_features=4, centers=2, cluster_std=6)

model = KNeighborsClassifier()
model.fit(X, y_true)

y_predict = model.predict(X)

t0 = time.time()
score = model.score(X, y_true)
print("Score :", score * 100, " - Time :", time.time() - t0)

t0 = time.time()
f1 = f1_score(y_true, y_predict)
print("F1 Score :", f1 * 100, " - Time :", time.time() - t0)
