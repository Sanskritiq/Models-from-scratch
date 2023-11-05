import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

from preprocessing import Loader, NGram, Utils

import os
import sys
path = os.path.dirname(os.path.abspath(__file__)) + "/"

class DecisionNode:
    def __init__(self, feature_index = None, left= None, right = None, threshold = None, label = None, depth = None) -> None:
        # for decision tree
        self.feature = feature_index
        self.left = left
        self.right = right
        self.threshold = threshold
        self.depth = depth
        # for leaf node
        self.label = label

class DecisionTree:
    def __init__(self, criteria = 'gini') -> None:
        self.root = None
        self.criteria = criteria
        if criteria!= 'misclass' and criteria != 'entropy' and criteria != 'gini':
            print("invalid criteria, setting to gini")
            self.criteria = 'gini'
        self.loss_function = self._misclassification if criteria == 'misclass' else self._entropy if criteria == 'entropy' else self._gini

        # initialize the tree
        # self.fit(loader)

    def fit(self, loader: Loader):
        # all features 
        features = list(loader.train_data[0].features.keys())
        self.root = self._build_tree(features, loader.train_data, 0)

    def _build_tree(self, features, data, level):
        print("level", level)
        node = DecisionNode()
        node.depth = level
        node.feature, node.threshold, node.label = self.select_best_feature(features, data)
        

        if node.label:
            return node # got a leaf node
        elif node.feature and node.threshold:
            left_data, right_data = self.split_data(data, node.feature, node.threshold)

            if len(left_data) == 0 or len(right_data) == 0:
                _, node.label = self._get_max_label(data)
                return node     

            new_features = features.copy()
            new_features.remove(node.feature)
            node.left = self._build_tree(new_features, left_data, level+1)
            node.right = self._build_tree(new_features, right_data, level+1)

            return node
        
        _, node.label = self._get_max_label(data)
        return node
    
    def split_data(self, data, feature, threshold):
        left_data = []
        right_data = []

        for d in data:
            if d.features[feature] < threshold:
                left_data.append(d)
            else:
                right_data.append(d)
        
        return left_data, right_data
        
    def _get_max_label(self, data):
        label_counts = {}
        for d in data:
            if d.label not in label_counts:
                label_counts[d.label] = 0
            label_counts[d.label] += 1
        
        return label_counts,max(label_counts, key = label_counts.get)

    def select_best_feature(self, features, data):
        
        label_counts , label = self._get_max_label(data)
        if len(label_counts) == 1:
            return None, None, label
        
        # print("features", len(features))

        best_feature = None
        best_threshold = None
        best_gain = None

        for feature in features:
            threshold, gain = self.select_best_threshold(data, feature)
            # print(best_gain, gain, threshold, feature) 
            if best_gain is None  or gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

        # if no feature is selected, then it is a leaf node, so return the label
        if best_feature is None:
            return None, None, max(label_counts, key = label_counts.get)
        else:
            return best_feature, best_threshold, None

    def select_best_threshold(self, data, feature):
        # sort the data according to the feature
        data.sort(key = lambda x: x.features[feature])
        # get the unique values of the feature
        values = list(set([d.features[feature] for d in data]))
        # get the mid points of the values
        mid_points = [(values[i] + values[i+1]) / 2 for i in range(len(values) - 1)]
        # get the best threshold
        if len(mid_points) == 0 :
            mid_points = values
        best_threshold = None
        best_gain = None
        for threshold in mid_points:
            left_data, right_data = self.split_data(data, feature, threshold)
            gain = self._information_gain(self.loss_function, data, left_data , right_data)
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return best_threshold, best_gain


    def _information_gain(self, loss_function, data, left_data, right_data):
        loss_before = loss_function(data)
        loss_left = loss_function(left_data)
        loss_right = loss_function(right_data)

        total = len(left_data) + len(right_data)
        weighted_loss = loss_left * ( len(left_data) / total ) + loss_right * ( len(right_data) / total )
        return loss_before - weighted_loss
    
    def test(self, loader):
        all_lables = set()
        for data in loader.test_data:
            all_lables.add(data.label)
        
        labels = list(all_lables)
        reverse_map = {labels[i]:i for i in range(len(labels))}

        matrix = [[0 for i in range(len(labels))] for j in range(len(labels))]

        i = 0
        for data in loader.test_data:
            x = self._predict(self.root, data)
            matrix[labels.index(data.label)][labels.index(x)] += 1

        max = 0
        for i in range(len(labels)):
            for j in range(len(labels)):
                if matrix[i][j] > max:
                    max = matrix[i][j]
        # normalize & printing the matrix
        for i in range(len(labels)):
            for j in range(len(labels)):
                print(f"{matrix[i][j]}\t", end="")
                matrix[i][j] /= max
            print()

        Utils.calculate_score(matrix, labels, reverse_map)

        matrix = np.array(matrix)
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        # Customize the plot as needed (e.g., labels, title, color map)
        plt.colorbar(label='Color Scale')
        plt.xticks(range(matrix.shape[1]), labels=labels)
        plt.yticks(range(matrix.shape[0]), labels=labels)
        # x and y labels
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Heatmap')
        # Show the plot
        plt.show()



    def _predict(self, node, data):
        if node is None:
            return None
        if node.label:
            return node.label
        elif node.feature and node.threshold:
            if data.features[node.feature] < node.threshold:
                return self._predict(node.left, data)
            else:
                return self._predict(node.right, data)
        else:
            return None

    def display(self, node):
        if node.left:
            print(" "*node.depth + f"if {node.feature} < {node.threshold}:")
            self.display(node.left)
        if node.right:
            print(" "*node.depth + f"else:")
            self.display(node.right)
        if node.label:
            print(" "*node.depth + f"return {node.label}")


    def _entropy(self, data):
        total_samples = len(data)
        label_counts = {}
        for d in data:
            if d.label not in label_counts:
                label_counts[d.label] = 0
            label_counts[d.label] += 1

        entropy = 0
        for label in label_counts:
            p = (label_counts[label] / total_samples)
            entropy -= p * math.log2(p)

        return entropy

    def _gini(self, data):
        total_samples = len(data)
        label_counts = {}
        for d in data:
            if d.label not in label_counts:
                label_counts[d.label] = 0
            label_counts[d.label] += 1
        
        gini = 1
        for label in label_counts:
            gini -= (label_counts[label] / total_samples) ** 2
        
        return gini

    def _misclassification(self, data):
        total_samples = len(data)
        label_counts = {}
        for d in data:
            if d.label not in label_counts:
                label_counts[d.label] = 0
            label_counts[d.label] += 1

        missclass = 1
        max = 0
        for label in label_counts:
            if label_counts[label] > max:
                max = label_counts[label]
        try: 
            missclass -= (max / total_samples)
        except:
            pass

        return missclass
    
    def save(self, title):
        with open(title, "wb") as modelfile:
            pickle.dump(self.root, modelfile)
        print("model saved")
    
    def load(self, title):
        with open(title, "rb") as modelfile:
            self.root = pickle.load(modelfile)

def checkQ4():
    l = Loader("datasets")
    ngrams = NGram(3, l, [500,300,200])
    l._extract_features(ngrams.grams, 3)
    l._extract_features_test(ngrams.grams, 3)

    # define decision tree
    tree1= DecisionTree('gini')
    tree2= DecisionTree('entropy')
    tree3= DecisionTree('misclass')

    # load tree
    tree1.load("DT_gini_hue.pkl")
    tree2.load("DT_entropy.pkl")
    tree3.load("DT_misclass.pkl")

    # test
    count = 0
    for data in l.test_data:
        x1 = tree1._predict(tree1.root, data)
        x2 = tree2._predict(tree2.root, data)
        x3 = tree3._predict(tree3.root, data)
        if x1 == x2 == x3:
            continue
        else:
            print(f"Label: {data.label}, gini_pred = {x1}, entropy_pred = {x2}, misclass_pred = {x3}")
            count += 1
    print(f"total {count} errors")


if __name__== '__main__':
    # load dataset and extract feature
    l = Loader("datasets")
    ngrams = NGram(3, l, [500,300,200])
    l._extract_features(ngrams.grams, 3)
    l._extract_features_test(ngrams.grams, 3)
    
    # gini, entropy, misclass
    tree = DecisionTree('misclass')
    # tree.fit(l)
    # tree.save("DT_misclass.pkl")
    tree.load("DT_entropy.pkl")
    # tree.display(tree.root)
    tree.test(l)
    # checkQ4()
    