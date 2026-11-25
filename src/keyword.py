import re
from collections import Counter
from math import log


def tokenize(text):
    """Split English and Bangla text into lowercase words."""
    return re.findall(r"[a-z0-9]+|[\u0980-\u09FF]+", text.lower())


class KeywordIndex:
    """A simple BM25 keyword search index."""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = [tokenize(doc) for doc in documents]
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avg_length = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths else 0
        )
        self.term_frequencies = [Counter(doc) for doc in self.documents]
        self.doc_frequency = Counter(
            word for doc in self.documents for word in set(doc)
        )

    def scores(self, query):
        """Return one BM25 score per document."""
        query_words = tokenize(query)
        total_docs = len(self.documents)
        scores = [0.0] * total_docs

        if not total_docs or not self.avg_length:
            return scores

        for word in query_words:
            df = self.doc_frequency.get(word, 0)
            if not df:
                continue

            idf = log(1 + (total_docs - df + 0.5) / (df + 0.5))

            for i, frequencies in enumerate(self.term_frequencies):
                tf = frequencies.get(word, 0)
                if not tf:
                    continue

                length = self.doc_lengths[i]
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * length / self.avg_length
                )
                scores[i] += idf * tf * (self.k1 + 1) / denominator

        return scores