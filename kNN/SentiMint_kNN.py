
import pickle

class emotionCalculator:
    def __init__(self):
        model_file = open("model.knn","rb")
        tfidf_file = open("corpus.tfidf","rb")

        self._model = pickle.load(model_file)
        self._vectorizer = pickle.load(tfidf_file)

        model_file.close()
        tfidf_file.close()

    def predict(self, text):
        tfidf_vector = self._vectorizer.transform([text])

        return self._model.predict(tfidf_vector)[0]
