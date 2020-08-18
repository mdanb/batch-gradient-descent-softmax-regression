import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets

iris = datasets.load_iris()

X = iris["data"][:, (2, 3)].T  # petal length, petal width
y = iris["target"]
X = np.r_[np.ones([1, len(X.T)]), X]

# X = np.array([[1,0,1],[1,1,0]])
# y = np.array([0,1,2])
# Shape of X: features x instances

def batch_grad_descent(X, y, lr, num_epochs, plot=True, val_percentage=0.1):
    num_instances = X.shape[1]
    num_features = X.shape[0]
    p = np.random.permutation(num_instances)
    X = (X.T[p]).T
    y = (y.T[p]).T
    k = len(np.unique(y))
    big_theta = np.random.rand(k, num_features)  # parameter matrix: classes x num_features
    y_one_hot = np.eye(k)[y]

    partition_threshold = int(np.ceil(num_instances * (1 - val_percentage)))
    X_train, X_val = X[:, 0:partition_threshold], X[:, partition_threshold:]
    y_train, y_val = y[0:partition_threshold], y[partition_threshold:]
    y_train_one_hot = y_one_hot[0:partition_threshold]
    # rows = np.arange(0, X_train.shape[1])

    exponentiate = lambda score: np.exp(score)
    vexponentiate = np.vectorize(exponentiate)
    total_costs = [0]
    cumulative_error = 0

    for epoch in range(num_epochs):
        s = X_train.T @ big_theta.T
        s_exp = vexponentiate(s)
        sums = np.sum(s_exp, axis=1).reshape(-1, 1)
        probs = s_exp / sums
        rows = np.arange(X_train.shape[1])

        if plot:
            actual_class_predicted_prob = probs[rows, y_train]
            cost = -1 / X_train.shape[1] * np.sum(np.log(actual_class_predicted_prob))
            cumulative_error += cost
            total_costs.append(cost + cumulative_error)

        for idx, column in enumerate((probs - y_train_one_hot).T):
            grad_k = (1 / X_train.shape[1] * np.sum(column.reshape(-1, 1) * X_train.T, axis=0))
            big_theta[idx] -= lr * grad_k

        print('Validation Accuracy:')
        s = X_val.T @ big_theta.T
        s_exp = vexponentiate(s)
        sums = np.sum(s_exp, axis=1).reshape(-1, 1)
        probs = s_exp / sums
        predicted_classes = np.argmax(probs, axis=1)
        print(sum(predicted_classes == y_val) / len(y_val))

    if plot:
        plt.plot(np.arange(0, num_epochs + 1), total_costs)

    return big_theta


print(batch_grad_descent(X, y, 0.01, 8000, val_percentage=0.10))