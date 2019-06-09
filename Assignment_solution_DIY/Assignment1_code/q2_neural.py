#coding=utf-8

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1 = data.dot(W1)+b1
    h = sigmoid(z1)
    z2 = h.dot(W2)+b2
    y_ = softmax(z2)
    cost = np.sum(-np.log(y_[labels==1]))/data.shape[0]
    ### END YOUR CODE

    # ### YOUR CODE HERE: backward propagation
    # theta1 = y_-labels
    # gradW2 = h.T.dot(theta1)
    # gradb2 = theta1
    # theta2 = theta1.dot(W2.T)
    # theta3 = theta2*(sigmoid_grad(sigmoid(z1)))
    # #theta3 = sigmoid_grad(sigmoid_grad(z1)).dot(theta2)
    # #gradW1 = theta3.dot(data.T)
    # gradW1 = data.T.dot(theta3)
    # gradb1 = theta3
    # ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    theta1 = (y_-labels)/data.shape[0]
    gradW2 = h.T.dot(theta1)
    gradb2 = np.sum(theta1,0)

    theta2 = theta1.dot(W2.T)
    theta3 = theta2*(sigmoid_grad(h))
    gradW1 = data.T.dot(theta3)
    gradb1 = np.sum(theta3,0)

    """
    总结：
        (1) 如果是按theta误差项来反向传播，W.T是乘在右边(dot)
        (2) 如果是对W求导，得出的系统应该是左乘(dot)
        (3) 如果是有激活函数的，则反向传播的对激活函数求导之后，按元素相乘(*)
    """
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()
