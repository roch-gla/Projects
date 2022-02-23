from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
import random
import pickle
import re
import nltk
from nltk.stem import PorterStemmer

__authors__ = ['Aurélien Houdbert', 'Gladys Roch']
__emails__ = ['aurelien.houdbert@student.ecp.fr','gladys.roch@supelec.fr']

nltk.download('stopwords')
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
ps = PorterStemmer()


def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            doc = re.sub(r'[^a-zA-Z\s]', '', l)
            doc = doc.lower()
            doc = doc.strip()
            tokens = wpt.tokenize(doc)
            filtered_tokens = [ps.stem(token) for token in tokens if (
                token not in stop_words) and (len(token) > 1)]
            sentences.append(filtered_tokens)
    return sentences


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=2, winSize=10, minCount=3):
        self.negativeRate = negativeRate
        self.nEmbed = nEmbed
        self.winSize = winSize
        self.minCount = minCount
        self.lr = 0.05

        # Preprocessing
        full_list_of_words = []
        for sentence in sentences:
            full_list_of_words += sentence

        vocab = []
        self.vocab_occ = {}
        for word in set(full_list_of_words):
            occurence = full_list_of_words.count(word)
            if occurence >= self.minCount:
                vocab.append(word)
                self.vocab_occ[word] = occurence

        #self.vocab = set(vocab)
        self.vocab = list(self.vocab_occ.keys())

        self.trainset = sentences
        self.w2id = {word: idx for idx, word in enumerate(self.vocab)}
        self.id2occ = {idx: self.vocab_occ[word]
                       for word, idx in self.w2id.items()}

        self.W = normalize(np.random.rand(len(self.vocab), nEmbed))
        self.C = normalize(np.random.rand(len(self.vocab), nEmbed))

        self.trainWords = 0
        self.accLoss = 0
        self.loss = []

    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""
        list_ids = [i for i in range(len(self.vocab)) if i not in list(omit)]
        #rand_ids = random.sample(list_ids, self.negativeRate)

        probas = np.array([self.id2occ[i] for i in range(
            len(self.vocab)) if i not in list(omit)])
        powers = np.power(probas, 0.75)
        probas = powers / np.sum(powers)

        rand_ids = np.random.choice(
            list_ids, self.negativeRate, replace=False, p=probas)

        return rand_ids

    def train(self):
        for counter, sentence in enumerate(self.trainset):
            sentence = list(filter(lambda word: word in self.vocab, sentence))

            for wpos, word in enumerate(sentence):
                wIdx = self.w2id[word]
                winsize = np.random.randint(self.winSize) + 1
                start = max(0, wpos - winsize)
                end = min(wpos + winsize + 1, len(sentence))

                for context_word in sentence[start:end]:
                    ctxtId = self.w2id[context_word]
                    if ctxtId == wIdx:
                        continue
                    negativeIds = self.sample({wIdx, ctxtId})
                    self.trainWord(wIdx, ctxtId, negativeIds)
                    self.trainWords += 1

            if counter % 1000 == 0:
                print(' > training %d of %d' % (counter, len(self.trainset)))
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0
                self.accLoss = 0.

    def trainWord(self, wordId, contextId, negativeIds):
        loss = 0
        X = self.W[wordId, :]
        Y = self.C[contextId, :]

        main_product = np.dot(X, Y)
        # compute derivative for Y
        #Ly = - X / (1 + np.exp(main_product))
        Ly = - X * expit(-main_product)

        # compute derivative for X
        #Lx = - Y / (1 + np.exp(main_product))
        Lx = - Y * expit(-main_product)

        loss += -np.log(expit(main_product))

        for negId in negativeIds:
            Z = self.C[negId, :]
            secondary_product = np.dot(X, Z)
            # Comput derivative for Z
            Lz = X * expit(secondary_product)
            # Update parameters for Z
            self.C[negId, :] = self.C[negId, :] - self.lr * Lz

            # Compute piece of derivative for X
            Lx += Z * expit(secondary_product)

            # Update Loss
            loss += -np.log(expit(-secondary_product))

        # Sum up all derivatives for X
        self.C[contextId, :] = self.C[contextId, :] - self.lr * Ly
        self.W[wordId, :] = self.W[wordId, :] - self.lr * Lx
        self.accLoss += loss

    def save(self, path):
        file = open(path, 'wb')
        pickle.dump(self, file)
        file.close()

    def similarity(self, word1, word2):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        word1 = ps.stem(word1)
        #print(word1, word1 in self.vocab)
        word2 = ps.stem(word2)
        #print(word2, word2 in self.vocab)

        if word1 in self.vocab:
            id_w1 = self.w2id[word1]
            #embed_word1 = (self.W[id_w1,:] + self.C[id_w1,:]) / 2
            embed_word1_W = self.W[id_w1, :]
            embed_word1_C = self.C[id_w1, :]
        else:
            #mean_embed = (self.W.mean(axis=0) + self.C.mean(axis=0)) / 2
            mean_embed_W = self.W.mean(axis=0)
            mean_embed_C = self.C.mean(axis=0)
            embed_word1_W = mean_embed_W
            embed_word1_C = mean_embed_C

        if word2 in self.vocab:
            id_w2 = self.w2id[word2]
            #embed_word2 = (self.W[id_w2,:] + self.C[id_w2,:]) / 2
            embed_word2_W = self.W[id_w2, :]
            embed_word2_C = self.C[id_w2, :]
        else:
            #mean_embed = (self.W.mean(axis=0) + self.C.mean(axis=0)) / 2
            mean_embed_W = self.W.mean(axis=0)
            mean_embed_C = self.C.mean(axis=0)
            embed_word2_W = mean_embed_W
            embed_word2_C = mean_embed_C

        #sim = expit(np.dot(embed_word1_W, embed_word2_W))
        #sim = expit(np.dot((embed_word1_W+embed_word1_C)/2, (embed_word2_W+embed_word2_C)/2))
        #sim = expit((np.dot(embed_word1_W, embed_word2_C) + np.dot(embed_word1_C, embed_word2_W))/2)
        #sim = np.dot(embed_word1_C, embed_word2_C) / (np.linalg.norm(embed_word1_C) * np.linalg.norm(embed_word2_C))
        sim = abs(np.dot((embed_word1_W + embed_word1_C)/2, (embed_word2_W + embed_word2_C)/2) / \
            (np.linalg.norm((embed_word1_W + embed_word1_C)/2)
             * np.linalg.norm((embed_word2_W + embed_word2_C)/2)))

        return sim

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            return pickle.load(file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text', help='path containing training data', required=True)
    parser.add_argument(
        '--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        epochs = 5
        for epoch in range(epochs):
            sg.train()
            print('Epoch [', epoch + 1, ':', epochs, '] done...')
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a, b, _ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a, b))
