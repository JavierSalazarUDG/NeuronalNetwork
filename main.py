import numpy as np
from RPM import *
# X = (hours sleeping, hours studying), y = test score of the student
X = np.array(([0, 0], [0, 1], [1, 0],[1,1]), dtype=float)
y = np.array(([0], [1], [1],[0]), dtype=float)

# scale units
# X = X/np.amax(X, axis=0) #maximum of X array
# y = y/100 # maximum test score is 100
NN = NeuralNetwork()
epochs = 50000
epoch = 0
presition = 0.0001
E = 1.0
while epoch < epochs and E > presition:

    E = np.mean(np.square(y - NN.feedForward(X)))
    NN.train(X, y)
    epoch = epoch +1

print("Epoca: " + str(epoch))

# for i in range(50000): #trains the NN 1000 times
#     if (i % 100 == 0):
#         print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
#     NN.train(X, y)
        
# print("Input: " + str(X))
# print("Actual Output: " + str(y))
# print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
# print("\n")
print("Predicted Output: " + str(NN.feedForward(X)))