#!/usr/bin/env python

from logging import raiseExceptions
import math
import numpy as np
import cupy as cp
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()


class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * q_freq * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * q_freq * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score.tolist()


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score.tolist()


## liping
## efficient implementation using cupy
class BM25Cupy(BM25Okapi):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _zero_cp(self):
        # using numpy instead of cupy for saving GPU memory and improving building speed
        return np.zeros(len(self.doc_len))

    def _initialize(self, corpus):
        nd = super()._initialize(corpus)
        # construct a dict: word -> cupy array of freq in each doc
        self.word_freq = defaultdict(self._zero_cp)  # initialze the cupy array value of each word as a zero array
        print("Building BM25 word frequencies")
        for i, doc_freq in tqdm(enumerate(self.doc_freqs)):
            for word, freq in doc_freq.items():
                self.word_freq[word][i] = freq
        return nd

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        super()._calc_idf(nd)
        self.idf["##pad"] = 0  # add idf for padding word

    def get_batch_scores(self, query_batch):
        """
        Calculate bm25 scores between a batch of queries and all docs
        """       
        max_len = max([len(query) for query in query_batch])
        query_padded = [query+["##pad"]*(max_len-len(query)) for query in query_batch]
        idf_array = cp.array([[self.idf.get(q, 0) for q in query] for query in query_padded])
        q_freq_array = cp.array([[cp.asarray(self.word_freq.get(q, cp.zeros(len(self.doc_len)))) for q in query] for query in query_padded])
        doc_len = cp.array(self.doc_len)
        score = ((cp.array(self.k1 + 1) * idf_array.reshape(-1, max_len, 1)) * q_freq_array)  \
                / (q_freq_array + cp.array(self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))

        return score.sum(1)

    def get_top_n_id(self, query_batch, n=5):
        top_n_ids = []
        scores_batch = self.get_batch_scores(query_batch)
        for scores in scores_batch:
            top_n_ids.append(cp.argsort(scores)[::-1][:n])
        return top_n_ids

    def get_top_n(self, query_batch, documents, n=5):
        top_n_ids = []
        scores_batch = self.get_batch_scores(query_batch)
        for scores in scores_batch:
            top_n_ids.append(cp.argsort(scores)[::-1][:n])
        return [[documents[i] for i in top_n_id] for top_n_id in top_n_ids]
        

class BM25Numpy(BM25Okapi):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _zero_np(self):
        return np.zeros(len(self.doc_len))

    def _initialize(self, corpus):
        nd = super()._initialize(corpus)
        # construct a dict: word -> cupy array of freq in each doc
        self.word_freq = defaultdict(self._zero_np)  # initialze the cupy array value of each word as a zero array
        for i, doc_freq in tqdm(enumerate(self.doc_freqs)):
            for word, freq in doc_freq.items():
                self.word_freq[word][i] = freq
        return nd

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        super()._calc_idf(nd)
        self.idf["##pad"] = 0  # add idf for padding word

    def get_batch_scores(self, query_batch):
        """
        Calculate bm25 scores between a batch of queries and all docs
        """       
        max_len = max([len(query) for query in query_batch])
        query_padded = [query+["##pad"]*(max_len-len(query)) for query in query_batch]
        idf_array = np.array([[self.idf.get(q, 0) for q in query] for query in query_padded])
        q_freq_array = np.array([[self.word_freq.get(q, np.zeros(len(self.doc_len))) for q in query] for query in query_padded])
        doc_len = np.array(self.doc_len)
        score = ((np.array(self.k1 + 1) * idf_array.reshape(-1, max_len, 1)) * q_freq_array)  \
                / (q_freq_array + np.array(self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))

        return score.sum(1)
