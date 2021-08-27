#Mahmoud Mohamed Amr 20176027
#Hady Raed 20175019
import numpy as nump
import matplotlib.pyplot as plot

data = nump.genfromtxt('./data.csv', delimiter=',', skip_header=1)
X = nump.array([data[:,1]]).T
Y = nump.array([data[:,2]]).T
X = nump.hstack([nump.ones((X.shape)), X])
thetas = nump.zeros((1, X.shape[1]))
alpha= 0.01
iterations= 1500

def gradientDescent(X, Y, thetas, alpha, iterations):
    cost_list = []
    for _ in range(0, iterations):
        thetas = thetas - (alpha * nump.dot((nump.dot(X, thetas.T) - Y).T, X)) / len(Y)
        cost_list.append(costFunc(X, Y, thetas))
    return thetas, cost_list

def costFunc(X, Y, thetas):
    cost = nump.sum(pow(nump.dot(X, thetas.T) - Y, 2)) / (len(Y) * 2)
    return cost

def predict(X,thetas):
    return nump.dot(X, thetas.T)

k = input("Enter the number to population to predict the profit: ")
x= nump.array([1, float(k)])
thetas, cost_list = gradientDescent(X,Y,thetas,alpha,iterations)
print('theta 0: ', thetas[0][0])
print('theta 1: ', thetas[0][1])
plot.plot(X[:, 1],Y, '.')
plot.plot(X[:, 1],nump.dot(X, thetas.T), '-')
plot.xlabel('population')
plot.ylabel('profit')
plot.show()
plot.plot(range(len(cost_list)), cost_list, '-')
plot.show()
print(predict(x,thetas))