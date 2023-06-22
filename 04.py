from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs

colors = ['red', 'green', 'blue']
X, y, centers = make_blobs(return_centers=True, cluster_std=10, center_box=[0,1])

print(X.shape)
print(y.shape)

plt.subplot(1, 2, 1)
plt.title("Real")
for i in range(len(X)):
    x_coord = X[i][0]
    y_coord = X[i][1]
    plt.scatter(x_coord, y_coord, c=colors[y[i]])

for x_coord, y_coord in centers:
    plt.scatter(x_coord, y_coord, c='black')

x_train,x_test , y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1, stratify=y)

model = LogisticRegression()
model.fit(x_train, y_train)
predict = model.predict(X)

plt.subplot(1, 2, 2)
for i in range(len(X)):
    x_coord = X[i][0]
    y_coord = X[i][1]
    plt.scatter(x_coord, y_coord, c=colors[predict[i]])

for x_coord, y_coord in centers:
    plt.scatter(x_coord, y_coord, c='black')
plt.title("Predict")

print(model.score(x_test,y_test) * 100)

plt.show()
