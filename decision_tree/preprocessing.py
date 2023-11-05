from nltk import pos_tag
START = '<start>'

class Data:
    def __init__(self, line) -> None:
        line = line.strip()
        self.label = line.split(":")[0]
        self.data = line.split(":")[1].split(" ")[:-1]
        self.length = len(self.data)
        self.pos = pos_tag(self.data)
        self.features = None



    def _get_features(self, freq_grams, n):
        features = {}
        features['length'] = self.length

        # n-gram
        for i in range(1, 1+n):
            p = [START]*(i-1) + self.data
            grams = {}
            for j in range(len(p) - i):
                _key = tuple(p[j:j+i])
                grams[_key] = 1

            for key in freq_grams[i]:
                if key in grams:
                    features[key] = 1
                else:
                    features[key] = 0

        # pos uni-gram for most frequest n grams 
        
        return features
    
class Loader:
    def __init__(self, directory, train_file = "train_5500.label", test_file = "TREC_10.label") -> None:
        # assuming file name to be train
        self.train_data = []
        self.test_data = []
        self.load(directory, train_file, test_file)

    def load(self, directory, train_file, test_file):
        for [file, l] in [[train_file, self.train_data], [test_file, self.test_data]]:
            f = open(f"{directory}/{file}", "r", encoding='ISO-8859-1')
            for line in f:
                l.append(Data(line))

    def _extract_features(self, freq_grams, n):
        for data in self.train_data:
            data.features = data._get_features(freq_grams, n)
    
    def _extract_features_test(self, freq_grams, n):
        for data in self.test_data:
            data.features = data._get_features(freq_grams, n)


class NGram:
    def __init__(self, n, loader, freq_caps) -> None:
        self.n = n
        self.grams = [None]
        self.freq = [None]

        # populate n gram upto n
        for i in range(1,n+1):
            self.update(loader, i, freq_cap=freq_caps[i-1])

    def update(self, loader, n, freq_cap):
        grams = {}
        for data in loader.train_data:
            p = [START]*(n-1) + data.data
            for i in range(len(p) - n):
                _key = tuple(p[i:i+n])
                if _key not in grams:
                    grams[_key] = 0
                grams[_key] += 1

        res = { key:value for key, value in sorted(grams.items(), key= lambda ele: ele[1], reverse= True)}
        freq_ngrams = []
        freq = []
        i = 0
        for key,value in res.items():
            freq_ngrams.append(key)
            freq.append(value)
            i +=1 
            if i > freq_cap:
                self.grams.append(freq_ngrams)
                self.freq.append(freq)
                return

class Utils:
    @staticmethod
    def calculate_score(matrix, labels, reverse_map):
        # precision
        precision = {}
        for label in labels:
            l = reverse_map[label]
            precision[l] = matrix[l][l] / sum(matrix[l])
        
        # recall
        recall = {}
        for label in labels:
            l = reverse_map[label]
            recall[l] = matrix[l][l] / sum([matrix[i][l] for i in range(len(labels))])
        
        # f1 score
        f1 = {}
        for label in labels:
            label = reverse_map[label]
            f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label])
        
        # print the score 
        for l in range(len(labels)):
            print(f"{labels[l]} : precision {precision[l]}, recall {recall[l]}, f1 score {f1[l]}")




if __name__ == '__main__':
    l = Loader("datasets")
    ngrams = NGram(3, l, [100,60,40])
    l._extract_features(ngrams.grams, 3)
    l._extract_features_test(ngrams.grams, 3)

