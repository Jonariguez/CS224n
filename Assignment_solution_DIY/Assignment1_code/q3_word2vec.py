#coding=utf-8

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    F = np.apply_along_axis(lambda x:np.sqrt(x.T.dot(x)),axis=1,arr=x)
    x /= F.reshape(x.shape[0],1)
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    """
    重申一下：
        predicted       对应作业中的 v
        target          对应作用中的 I(y==1)
        outputVectors   对应作用中的 u (注意作业上的U是[u1,u2,..uW],维度为d*W,而这里是W*d的)
        gradPred        对应作用中的 dJ/dv_c
        grad            对应作用中的 dJ/du_w
    """
    v = predicted       # d*1
    u = outputVectors   # d*W
    y_ = softmax(u.dot(v))

    # cost是交叉熵
    cost = -np.log(y_[target])

    Y = y_.copy()
    Y[target] -= 1.0

    gradPred = u.T.dot(Y)
    grad     = np.outer(Y,v)

    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    v = predicted
    u = outputVectors
    # indices[0]里面保存的是target，即正确的背景词，对用作用中的o，而indices[1:]都是噪声词，通过负采样得到的
    grad = np.zeros(u.shape)
    gradPred = np.zeros(v.shape)
    cost = 0

    # 先算正例的损失和梯度
    val = sigmoid(u[target].dot(v))-1
    cost -= np.log(val+1)

    gradPred += val*u[target]
    grad[target] = val*v

    # 然后再算负例，注意利用了:1-sigmoid(-x)=sigmoid(x)
    for samp in indices[1:]:
        val = sigmoid(u[samp].dot(v))
        gradPred += val*u[samp]
        grad[samp] += val*v
        cost -= np.log(1-val)

    # cost = -np.log(sigmoid(u[indices[0]].dot(v)))-np.sum(np.log(sigmoid(-u[indices[1:]].dot(v))),0)
    #
    # rea_idx = target
    # neg_idx = indices[1:]
    # gradPred = sigmoid(sigmoid(u[rea_idx].dot(v))-1)*u[rea_idx]+np.sum((1-sigmoid(-u[neg_idx].dot(v)))*u[neg_idx],0)
    # grad =
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """
    """
    currrentWord        中心词
    contextWords        背景词
    inputVectors        初始化的词向量v
    outputVectors       训练好的词向量u
    """
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    center_id = tokens[currentWord]
    # 拿到中心词的中心词向量，对应作业中的v_c
    v = inputVectors[center_id]

    for context in contextWords:
        context_id = tokens[context]
        # 真正的背景词下标为context_id --> target
        c_cost,c_gradin,c_gradout = word2vecCostAndGradient(v,context_id,outputVectors,dataset)
        cost += c_cost
        # 对于v_c的梯度要增加在center_id上，而不是gradIn += c_gradin
        gradIn[center_id] += c_gradin
        gradOut += c_gradout
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    # CBOW模型是根据周围的词来预测中间的词，即用多个背景词来推测一个中心词，和skip-gram一样的
    # 是：每次都是推测一个次.(skip-gram是每次推测一个，然后推测多次，而CBOW是就推测一次)
    # 而和skip-gram不一样的是：skip-gram是用一个词来推测，而CBOW是用多个词。但是我们之前
    # 实现的代价函数就是L(一个中心词,一个背景词)。所以CBOW模型的多个词就要压缩成一个词，也就是
    # 用这多个词向量的和作为一个背景词来推测中心词。
    center_ids = [tokens[w] for w in contextWords]
    v = np.sum(inputVectors[center_ids],0)
    target = tokens[currentWord]
    cost,temp_gradin,gradOut = word2vecCostAndGradient(v,target,outputVectors,dataset)
    # 根据作业中的求导，只对背景词的梯度进行修改
    for i in center_ids:
        gradIn[i] += temp_gradin
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()