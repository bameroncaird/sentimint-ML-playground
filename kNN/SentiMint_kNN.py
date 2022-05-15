import pickle

class sentiMint_kNN:
    def __init__(self):
        m = open("model.knn","rb")
        v = open("vocab.tfidf","rb")
        self._kNN_alg = pickle.load(m)
        self._vectorizer = pickle.load(v)
        m.close()
        v.close()

    def predict(self, text):
        tfidf_vector = self._vectorizer.transform([text])
        return self._kNN_alg.predict(tfidf_vector)[0]
