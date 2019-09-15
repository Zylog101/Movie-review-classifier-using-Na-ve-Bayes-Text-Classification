import sys
from math import log, exp

import matplotlib.pyplot as plt
import numpy as np

from Eval import Eval
from imdb import IMDBdata


# calculating the total occurence of documnt words in the category
def count_of_y_in_x(X, indices, z):
    rows = X.tocsr()[indices, :]
    sum = 0
    column = rows.tocsc()[:, z[1]]

    sum += column.sum().item(0)
    return sum


class NaiveBayes:

    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data

        self.vocab_len = data.vocab
        self.Train(data.X,data.Y)

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    # additional dictionary composing total occurrences of a word for positive and negative category is formed
    def Train(self, X, Y):

        self.positive_indices = np.argwhere(Y == 1.0).flatten()
        self.negative_indices = np.argwhere(Y == -1.0).flatten()

        self.count_positive = self.CountValidWords(X, self.positive_indices)
        self.count_negative = self.CountValidWords(X, self.negative_indices)

        self.num_positive_reviews = self.positive_indices.__len__()
        self.num_negative_reviews = self.negative_indices.__len__()

        self.total_positive_words = self.SumWordsInRows(X, self.positive_indices).item(0)
        self.total_negative_words = self.SumWordsInRows(X, self.negative_indices).item(0)

        self.P_positive = self.num_positive_reviews/(self.num_positive_reviews+self.num_negative_reviews)
        self.P_negative = self.num_negative_reviews/(self.num_positive_reviews+self.num_negative_reviews)

        self.deno_pos = self.total_positive_words + self.ALPHA
        self.deno_neg = self.total_negative_words + self.ALPHA

        # dictionary of occurrences of specific words in respecitive categories
        self.dictpos = {}
        self.dictneg = {}

        Pos_rows = X.tocsr()[self.positive_indices, :]
        Pos_rows = Pos_rows.tocsc()
        neg_rows = X.tocsr()[self.negative_indices, :]
        neg_rows = neg_rows.tocsc()
        for column in range(X.shape[1]):
            self.dictpos.update({column : Pos_rows.getcol(column).sum().item(0)})
            self.dictneg.update({column : neg_rows.getcol(column).sum().item(0)})
        return

    # calculating P(W|Y=y)
    def count_prob_of_w_and_y(self, row_test_x, indices, word_columns, den, dicti):
        sum_pos = 0
        for w in range(word_columns.shape[0]):
            word_freq_count = dicti.get(word_columns[w])
            sum_pos += (log((word_freq_count + (self.ALPHA * 0.5)) / den)) * row_test_x[0, word_columns[w]]
        return sum_pos

    # Those unique words whose frequencies are non zeros are being summed up for each review
    def CountValidWords(self, X, indices):
        row = 0
        for index in indices:
            row += X.getrow(index).count_nonzero()
        return row

    def SumWordsInRows(self, X, indices):
        count = 0
        for index in indices:
            count += X.getrow(index).sum(axis=1)
        return count

    # Predict labels for instances in X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X, threshold):

        pred_labels = []
        positive_log = log(self.P_positive)
        negitive_log = log(self.P_negative)
        score = False
        sh = X.shape[0]
        for i in range(sh):
            z = X[i].nonzero()
            sum_pos = 0
            sum_neg = 0

            sum_pos = self.count_prob_of_w_and_y(X[i], self.positive_indices, z[1], self.deno_pos, self.dictpos) + positive_log

            sum_neg = self.count_prob_of_w_and_y(X[i], self.negative_indices, z[1], self.deno_neg, self.dictneg) + negitive_log
            if(exp(sum_pos) > threshold) and sum_pos > sum_neg:
                score = True
            else:
                score = False

            if score:            # Predict positive
                pred_labels.append(1.0)
            else:               # Predict negative
                pred_labels.append(-1.0)
        
        return pred_labels

    def LogSum(self, logx, logy):   

        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))


    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
    def PredictProb(self, test, indexes):
        sum_positive = 0
        sum_negative = 0
        positive_log = log(self.P_positive)
        negitive_log = log(self.P_negative)
        for i in indexes:
            z = test.X[i].nonzero()
            sum_pos = 0
            sum_neg = 0
            sum_pos = self.count_prob_of_w_and_y(test.X[i], self.positive_indices, z[1], self.deno_pos, self.dictpos) + positive_log

            sum_neg = self.count_prob_of_w_and_y(test.X[i], self.negative_indices, z[1], self.deno_neg, self.dictneg) + negitive_log

            deno = self.LogSum(sum_pos, sum_neg)
            sum_positive = sum_pos - deno
            sum_negative = sum_neg - deno

            predicted_prob_positive = exp(sum_positive)
            predicted_prob_negative = exp(sum_negative)
            
            if predicted_prob_positive > predicted_prob_negative:
                predicted_label = 1.0
            else:
                predicted_label = -1.0

            print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i])

        precision_values = self.EvalPrecision(self.positive_indices, test)
        recall_values = self.EvalRecall(self.positive_indices, test)
        self.plot_prec_recall_graph(precision_values, recall_values)

    def plot_prec_recall_graph(self, precision_values, recall_values):

        print(precision_values.items())
        print(recall_values.items())

        # line 1 points
        x1 = list(precision_values.keys())
        y1 = list(precision_values.values())
        # plotting the line 1 points
        plt.plot(x1, y1, label="Precision")

        # line 2 points
        x2 = list(recall_values.keys())
        y2 = list(recall_values.values())
        # plotting the line 2 points
        plt.plot(x2, y2, label="Recall")

        # naming the x axis
        plt.xlabel('x - axis (threshold)')
        # naming the y axis
        plt.ylabel('y - axis (occurrence)')
        # giving a title to my graph
        plt.title('precision/recall graph!')

        # show a legend on the plot
        plt.legend()

        # function to show the plot
        plt.show()

    # false Positive
    def EvalPrecision(self, index, test):
        precision_count = 0
        tp = 0
        dicti = {0.25: 0, 0.15: 0, 0.1: 0, 0.05: 0}
        for k, v in dicti.items():
            precision_count = 0
            tp = 0
            predicted_label = self.PredictLabel(test.X.tocsr()[index, :], k)
            for label in range(predicted_label.__len__()):
                if predicted_label[label] == 1.0 and test.Y[label] == -1.0:
                    precision_count = precision_count+1
                elif predicted_label[label] == test.Y[label]:
                    tp = tp + 1
            dicti[k] = tp/(tp + precision_count)
        return dicti

    # false Negative
    def EvalRecall(self, index, test):
        recall_count = 0
        tp = 0
        dicti = {0.25: 0, 0.15: 0, 0.1: 0, 0.05: 0}
        for k, v in dicti.items():
            recall_count = 0
            tp = 0
            predicted_label = self.PredictLabel(test.X.tocsr()[index, :], k)
            for label in range(predicted_label.__len__()):
                if predicted_label[label] == -1.0 and test.Y[label] == 1.0:
                    recall_count = recall_count+1
                elif predicted_label[label] == test.Y[label]:
                    tp = tp + 1
                    dicti[k] = tp / (tp + recall_count)
        return dicti

    # Evaluate performance on test data 
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X, 0.25)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()

    def print_twenty_most_pos_neg(self):
        positive_log = log(self.P_positive)
        negitive_log = log(self.P_negative)

        # sorted by values i.e in this case word frequencies
        sorted_pos_list = sorted(self.dictpos.items(), key=lambda t: t[1])
        sorted_neg_list = sorted(self.dictneg.items(), key=lambda t: t[1])

        out_str = ""
        out_str = out_str.__add__("20 most frequent positive words in vocab\n")
        for i in range(sorted_pos_list.__len__()-1, sorted_pos_list.__len__()-21, -1):
            (k, v) = sorted_pos_list[i]
            word = self.data.vocab.GetWord(k)
            weight = exp((log((v + (self.ALPHA * 0.5)) / self.deno_pos))+positive_log)
            out_str = out_str.__add__(" "+word+" "+weight.__str__())

        out_str = out_str.__add__("\n20 most frequent negative words in vocab\n")
        for i in range(sorted_neg_list.__len__()-1, sorted_pos_list.__len__()-21, -1):
            (k, v) = sorted_neg_list[i]
            word = self.data.vocab.GetWord(k)
            weight = exp((log((v + (self.ALPHA * 0.5)) / self.deno_neg)) + negitive_log)
            out_str = out_str.__add__(" " + word + " " + weight.__str__())
        print(out_str)


if __name__ == "__main__":
    
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Evaluating")
    print("Test Accuracy: ", nb.Eval(testdata))
    print("Predicting Probability of first 10 reviews")
    nb.PredictProb(testdata,np.array([1,2,3,4,5,6,7,8,9,10]))

    print("calculating 20 mosts postive and negative words in the dictionary ")
    nb.print_twenty_most_pos_neg()

