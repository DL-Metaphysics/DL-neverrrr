import numpy as np
import matplotlib.pyplot as plt


# 数据集大小
m = 30....
#迭代次数
iterations = 10000
# 学习率
alpha = 0.001
X0 = np.ones((m, 1))
X1 = np.array([1.1, 1.3, 1.5, 2, 2.2, 2.9, 3, 3.2, 3.2, 3.7, 3.9, 4, 4,
               4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6, 6.8, 7.1, 7.9, 8.2, 8.7,
               9, 9.5, 9.6, 10.3, 10.5]).reshape(m, 1)
X = np.hstack((X0, X1))
print(X[6])
y = np.array([39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218,
              55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940, 91738,
              98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872

]).reshape(m, 1)

def graFun_Random(theta, X, y, temp):
    diff = np.dot(X[temp], theta) - y[temp]
    return diff*X[temp]

def gradient_random(X, y, m):
    theta = np.array([100,10000]).reshape(2,1)#定义二维数组theta，其中包括theta0，theta1
    history_theta00 = np.zeros(iterations)
    history_theta11 = np.zeros(iterations)
    for i in range(iterations):
        temp = np.random.randint(0,m-1)
        gradient = graFun_Random(theta, X, y, temp)
        old = theta[1]
        history_theta00[i] = theta[0]
        history_theta11[i] = theta[1]
        theta[0] = theta[0] - alpha*gradient[0]
        theta[1] = theta[1] - alpha * gradient[1]

    print('theta', theta)
    return theta, history_theta00, history_theta11



def gradient_function(theta, X, y):
    '''求函数的梯度'''
    diff = np.dot(X, theta) - y #dot函数求积，此处求预测值与真实值的差即误差函数 m*1
    return np.dot(np.transpose(X), diff)

def gradient_descent(X, y, alpha,iterations):
    '''
    批量梯度下降
    alpha：学习率
    iterations：迭代次数
    '''
    theta = np.array([100, 100000]).reshape(2, 1)
    gradient = gradient_function(theta, X, y)

    #while(1000):
    history_theta0 = np.zeros(iterations)
    history_theta1 = np.zeros(iterations)
    for i in range(iterations):
        theta_old = theta
        #print('545454', theta)
        theta = theta - alpha * gradient
        # print('gradient', gradient)
        history_theta0[i] = theta[0]
        history_theta1[i] = theta[1]
        print('theta0 = ', theta[0], 'theta1 = ', theta[1])
        gradient = gradient_function(theta, X, y)

    return theta, history_theta0, history_theta1

optimal, history_theta0, history_theta1 = gradient_descent(X, y, alpha,iterations)
print('optimal:', optimal)
optimal1, history_theta00, history_theta11 = gradient_random(X, y , m)
plt.ylabel('J(Theta)')
plt.xlabel('Iterations')
plt.subplot(121)
plt.plot(range(iterations), history_theta0, 'b.', range(iterations), history_theta00, 'c.')
plt.subplot(122)
plt.plot(range(iterations),history_theta1, 'c.', range(iterations),history_theta11, 'b.')
plt.show()
