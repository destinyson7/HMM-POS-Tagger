import re
import numpy
from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np


class ViterbiDecoder:

    def __init__(self):
        self.tags = []
        self.transition_prob = defaultdict(dict)
        self.emission_prob = defaultdict(dict)

        self.extract()
        self.take_input()

    def extract(self):
        with open("./hmmmodel.txt", "r") as f:

            flag = 0

            for line in f:
                cur = line.strip()
                cur = re.sub('\s+', ' ', cur)

                if cur:
                    if cur.split(":")[0] == "Tags":
                        self.tags = cur.split(":")[1].strip().split(" ")
                        # print("Tags are", self.tags)

                    elif cur.split(":")[0] == "Transition Probability":
                        flag = 1

                    elif cur.split(":")[0] == "Emission Probability":
                        flag = 2

                    elif flag == 1:
                        cur = cur.split(" ")
                        # print(cur)
                        if cur[0] in self.transition_prob.keys() and cur[2] in self.transition_prob[cur[0]].keys():
                            # print("transition", cur[0], cur[2])
                            self.transition_prob[cur[0]
                                                 ][cur[2]] += float(cur[4])

                        else:
                            self.transition_prob[cur[0]
                                                 ][cur[2]] = float(cur[4])

                    elif flag == 2:
                        cur = cur.split(" ")
                        word = cur[1].split("(")[1].split("|")[0].lower()
                        tag = cur[1].split(")")[0].split("|")[1]
                        probability = cur[3]

                        if word in self.emission_prob.keys() and tag in self.emission_prob[word].keys():
                            # print("emission", word)
                            self.emission_prob[word][tag] += float(probability)

                        else:
                            self.emission_prob[word][tag] = float(probability)

    def q(self, v, u):
        if u in self.transition_prob.keys() and v in self.transition_prob[u].keys():
            return self.transition_prob[u][v]/sum(self.transition_prob[u].values())
            # return self.transition_prob[u][v]

        return 0

    def e(self, word, v):
        if word in self.emission_prob.keys() and v in self.emission_prob[word].keys():
            return self.emission_prob[word][v]/sum(self.emission_prob[word].values())
            # return self.emission_prob[word][v]

        return 0

    def Viterbi(self, input_sentence):
        input_sentence = "$ " + input_sentence
        input_sentence = re.sub('\s+', ' ', input_sentence)
        input_sentence = input_sentence.strip()

        tokenized_words = word_tokenize(input_sentence)

        n = len(tokenized_words) - 1

        S = {}
        S[0] = ["Begin"]

        for index in range(1, n+1):
            S[index] = self.tags

        dp = defaultdict(dict)
        parent = defaultdict(dict)

        dp[0]["Begin"] = 1

        for k in range(1, n+1):
            word = tokenized_words[k].lower()

            for v in S[k]:
                # print(k, S[k])
                probs = np.array(
                    [(dp[k-1][u] * self.q(v, u) * self.e(word, v)) for u in S[k-1]])

                dp[k][v] = np.max(probs)
                parent[k][v] = S[k-1][np.argmax(probs)]

        y = ["#" for k in range(0, n+1)]
        y[n] = S[n][np.argmax([dp[n][v] for v in S[n]])]

        # print(y)

        for k in range(n-1, 0, -1):
            # print(k)
            y[k] = parent[k+1][y[k+1]]

        if all([dp[n][v] == 0 for v in S[n]]):
            print("One or more words not present in corpus")

        else:
            print(" ".join(y[1:]))

    def take_input(self):
        # print("input:")
        sentence = input()
        # print(sentence)
        self.Viterbi(sentence)


if __name__ == "__main__":
    ViterbiDecoder()
