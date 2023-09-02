import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet = True)
nltk.download('punkt', quiet = True)
stop_words = set(stopwords.words('english'))

class StanfordSentiment:
    def __init__(self, path = './datasetSentences.txt'):
        self.data_set_path = path
        self.sentences = []

    def preprocess_text(self, sentence):
        out = nltk.word_tokenize(sentence)
        out = [x.lower() for x in out]
        out = [x for x in out if x not in stop_words]
        return out

    def build_sentences(self):
        df = pd.read_csv(self.data_set_path, sep = '\t')
        df['sentence'] = df['sentence'].map(self.preprocess_text)
        self.sentences = df['sentence'].tolist()

    def get_dataset(self):
        self.build_sentences()
        return self.sentences
