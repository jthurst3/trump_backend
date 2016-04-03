import numpy as np

class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100,bptt_truncate=4):
        # instance variables

        # number of words
        self.word_dim = word_dim
        # size of hidden layer
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # random initialization
        self.U = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim, hidden_dim))


    # forward propagation
    def forward_prop(self, x):
        # number of time steps
        T = len(x)
        # need to save all hidden states
        s = np.zeros((T+1, self.hidden_dim))
        # why do we need this???
        s[-1] = np.zeros(self.hidden_dim)
        # outputs at each time step
        o = np.zeros((T, self.word_dim))
        # for every time step
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        # return the output and the hidden units
        return [o,s]

    # prediction
    def predict(self, x):
        # perform forward prop, return index of highest score
        o, s = self.forward_prop(x)
        return np.argmax(o, axis=1)


def main():
    pass

if __name__ == '__main__':
    main()
