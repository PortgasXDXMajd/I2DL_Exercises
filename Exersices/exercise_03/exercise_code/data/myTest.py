import numpy as np;


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


x = np.random.permutation(50)


list = list(chunks(x,3))

for arr in list:
    print(arr)