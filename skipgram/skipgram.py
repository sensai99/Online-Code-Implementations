import numpy as np
from collections import defaultdict

class skipgram:

    def __init__(self, corpus, window_size = 1, embedding_size = 100, num_epochs = 1, learning_rate = 0.01):
        self.corpus = corpus

        self.word_freq = defaultdict(int)
        self.word_index = {}
        self.ind2word = {}
        self.dataset = []
        self.construct_dataset()
        
        self.vocab_size = len(self.word_index)
        
        # hyperparameters
        self.window_size = window_size
        self.output_layer_size = self.input_layer_size = self.vocab_size
        self.embedding_size = embedding_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # parameters
        self.word_weights = np.random.uniform(-1, 1, (self.vocab_size, self.embedding_size))
        self.context_weights = np.random.uniform(-1, 1, (self.embedding_size, self.output_layer_size))
    
    def construct_dataset(self):
        for sentence in self.corpus:
            for i, word in enumerate(sentence):
                self.word_freq[word] += 1
                self.dataset.append([word] + sentence[i + 1 : i + 3] + sentence[i - 2 : i])
        
        self.word_index = {word: i for i, word in enumerate(self.word_freq.keys())}
        self.ind2word = {self.word_index[word]: word for word in self.word_index.keys()}

    def word2onehot(self, word):
        one_hot = np.zeros(self.vocab_size)
        one_hot[self.word_index[word]] = 1
        return one_hot
    
    def get_negative_samples(self):
        negative_sample_index = np.random.randint(0, len(self.word_index))
        return [self.ind2word[negative_sample_index]]

    def train(self):
        for epoch in range(self.num_epochs):
            print("Epoch: ", epoch)
            for word_batch in self.dataset:
                target_word = word_batch[0]
                print("-- Training on word: ", target_word)
                one_hot_word = self.word2onehot(target_word)
                for context_word in word_batch[1:]:
                    negative_samples = self.get_negative_samples()
                    h, word_similarities = self.forward_pass(one_hot_word, context_word, negative_samples)
                    loss = self.compute_loss(word_similarities)
                    print("---- Loss: ", loss)
                    self.backward_propagation(h, word_similarities, target_word, context_word, negative_samples)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def forward_pass(self, one_hot_word, context_word, negative_samples):
        h = np.dot(self.word_weights.T, one_hot_word)
        word_similarites = []
        word_similarites.append(np.dot(self.context_weights.T[self.word_index[context_word]], h))
        for negative_sample in negative_samples[1:]:
            word_similarites.append(np.log(self.sigmoid(-np.dot(self.context_weights.T[self.word_index[negative_sample]], h))))
        return h, word_similarites

    # Reference - https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling
    def compute_gradients(self, h, word_similarities, context_word,  negative_samples):
        target_word_gradients = (self.sigmoid(word_similarities[0]) - 1) * self.context_weights.T[self.word_index[context_word]]
        i = 1
        for negative_sample in enumerate(negative_samples):
            target_word_gradients += self.sigmoid(word_similarities[i]) * self.context_weights.T[self.word_index[negative_sample]]
            i += 1
        
        context_word_gradients = (self.sigmoid(word_similarities[0]) - 1) * h
        negative_samples_gradients = []

        i = 1
        for negative_sample in enumerate(negative_samples):
            negative_samples_gradients.append(self.sigmoid(word_similarities[i]) * h)
            i += 1
        
        return target_word_gradients, context_word_gradients, negative_samples_gradients
    
    def compute_loss(self, word_similarities):
        loss = np.log(self.sigmoid(word_similarities[0]))
        loss += np.sum(np.log(self.sigmoid(-1 * word_similarities[1:])))
        return -loss
    
    def backward_propagation(self, h, word_similarities, target_word, context_word, negative_samples):
        target_word_gradients, context_word_gradients, negative_samples_gradients = self.compute_gradients(h, word_similarities, context_word, negative_samples)
        
        self.word_weights[self.word_index[target_word]] -= self.learning_rate * target_word_gradients
        self.context_weights[:, self.word_index[context_word]] -= self.learning_rate * context_word_gradients
        for i, negative_sample in enumerate(negative_samples):
            self.context_weights[:, self.word_index[negative_sample]] -= self.learning_rate * negative_samples_gradients[i]


if __name__ ==  '__main__':
    sampleCorpus = [["quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]]
    word2vec = skipgram(corpus = sampleCorpus)
    print(word2vec.dataset)
    word2vec.train()