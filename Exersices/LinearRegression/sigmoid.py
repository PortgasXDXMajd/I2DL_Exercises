import numpy as np
import matplotlib.pyplot as plt


def Cross_Entropy(y_hat, y):
    if y == 1:
      return -np.log(y_hat)
    else:
      return -np.log(1 - y_hat)
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
 
def derivative_Cross_Entropy(y_hat, y):
    if y == 1:
      return -1/y_hat
    else:
      return 1 / (1 - y_hat)
 
def derivative_sigmoid(x):
    return x*(1-x)
 
area = 200
N = 500
mu, sigma = 0, 1 # mean and standard deviation
x_0 = np.random.normal(mu, sigma, N)
x_1 = np.random.normal(mu + 5, sigma*0.50, N)
X = [x_0, x_1]
Y = [np.zeros_like(x_0), np.ones_like(x_1)]
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape[0]*X.shape[1])
Y = Y.reshape(Y.shape[0]*Y.shape[1])

W = np.random.uniform(low=-0.01, high=0.01, size=(1,))
W_0 = np.random.uniform(low=-0.01, high=0.01, size=(1,))
 
Epoch = 120
eta = 0.001
# Start training
subplot_counter = 1
E = []
for ep in range(Epoch):
    # Shuffle the train_data X and its Labels Y
    random_index = np.arange(X.shape[0])
    np.random.shuffle(random_index)
    e = []
    for i in random_index:
        Z = W*X[i] + W_0
        Y_hat = sigmoid(Z)
        # Compute the Error
        e.append(Cross_Entropy(Y_hat, Y[i]))
        # Compute gradients
        dEdW = derivative_Cross_Entropy(Y_hat, Y[i])*derivative_sigmoid(Y_hat)*X[i]
        dEdW_0 = derivative_Cross_Entropy(Y_hat, Y[i])*derivative_sigmoid(Y_hat)
        # Update the parameter
        print(dEdW.shape)
        print("===")
        W = W - eta*dEdW
        W_0 = W_0 - eta*dEdW_0
    if ep % 20 == 0:
        plt.subplot(2, 3, subplot_counter)
        plt.scatter(x_0, np.zeros_like(x_0), s=area, c='r', alpha=0.5, label="Class 0")
        plt.scatter(x_1, np.zeros_like(x_1), s=area, c='c', alpha=0.5, label="Class 1")
        minimum = np.min(X)
        maximum = np.max(X)
        s = np.arange(minimum, maximum, 0.01)
        plt.plot(s, W * s + W_0, '-k', label="z")
        plt.plot(s, sigmoid(W * s + W_0), '+b', label="y_hat")
        plt.legend()
        plt.ylim(-0.2, 1.25)
        plt.grid()
        plt.title("Epoch Number= %d" % ep + ", Error= %.3f" % np.mean(e))
        subplot_counter += 1
    E.append(np.mean(e))
plt.figure()
plt.grid()
plt.xlabel("Epochs During Training")
plt.ylabel("Binary Cross-Entropy Error")
plt.plot(E, c='r')
plt.show()