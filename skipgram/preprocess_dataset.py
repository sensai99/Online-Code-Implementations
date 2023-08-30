class StanfordSentiment:
    def __init__(self, path = "./datasetSentences.txt"):
        self.data_set_path = path
        self.sentences = []

    def build_sentences(self):
        with open(self.data_set_path, 'r') as f:
            i = 0
            for line in f:
                if i == 0:
                    i += 1
                    continue
                # print(line)
                sentence = line.strip().split()[1:]
                self.sentences.append([word.lower() for word in sentence])
                i += 1
    
    def get_dataset(self):
        self.build_sentences()
        return self.sentences
