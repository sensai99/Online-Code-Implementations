import numpy as np
import matplotlib.pyplot as plt
from skipgram import skipgram
from preprocess_dataset import StanfordSentiment

if __name__ == '__main__':
    
    # generate dataset
    dataset = StanfordSentiment(path = './test.txt')
    sample_corpus = dataset.get_dataset()

    # train word2vec
    word2vec = skipgram(corpus = sample_corpus, window_size = 2, negative_samples_count = 1, embedding_size = 4, num_epochs = 10000, learning_rate = 0.005)
    word2vec.train()

    word_index = word2vec.get_word_index()
    word_freq = word2vec.get_word_freq()

    most_freq_words = sorted(word_freq.items(), key = lambda x: x[1], reverse = True)[:10]
    words = [word for (word, freq) in most_freq_words]
    word_indices = [word_index[word] for word, freq in most_freq_words]

    word_vectors = word2vec.get_word_vectors()
    word_vectors = word_vectors[word_indices, :]

    temp = (word_vectors - np.mean(word_vectors, axis=0))
    covariance = 1.0 / len(word_indices) * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2])

    for i in range(len(words)):
        plt.text(coord[i,0], coord[i,1], words[i],
            bbox=dict(facecolor='green', alpha=0.1))

    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

    plt.savefig('word_vectors.png')