import numpy as np

def sigmoid(x):
    out = 1 / (1 + np.exp(-x))
    return out

def derivative_sigmoid(x):
    return x * (1 - x)

n = 100
noFeatures = 5


X = np.random.randint(100, size=(n,noFeatures))
W = np.random.randint(100, size=(noFeatures,1))
Y = sigmoid(np.matmul(X, W))
# x_out = np.random.randint(50, size=(n,noFeatures))
# # y_train = np.random.randint(100, size=(n,1))

# # M = np.ones(shape=(noFeatures, 1))
# # B = 1
# #print(M)
# # features = np.array([5,1,-1])
# # print(np.dot(x_train ,M))
# # y_pre = np.dot(x_train ,M) + B
# # print(y_pre)
# # error = y_pre - y_train

# #print(np.sum(np.abs(error)) / n)  # L1 Loss Function

# # print(features.shape)

# # print(matrix * features)



print(Y)

print(derivative_sigmoid(Y))
