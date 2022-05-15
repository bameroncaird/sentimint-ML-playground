import pandas
import numpy
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as kNN



POSNEG = {"negative": -1, "neutral": 0, "positive": 1}


def split_data(data, test_portion):
    training_data, testing_data = train_test_split(data, test_size=test_portion)
    training_data["sentiment"] = training_data["sentiment"].map(POS_NEG_MAPPING)
    testing_data["sentiment"] = testing_data["sentiment"].map(POS_NEG_MAPPING)

    training_data.fillna('',inplace=True)
    testing_data.fillna('',inplace=True)
    
    return [training_data, testing_data]





def train(file_name):
    data = pandas.read_csv(file_name)[["text","sentiment"]]
    [training_data, testing_data] = split_data(data, 0.2)
    tfidf = TfidfVectorizer()
    tfidf.fit(training_data["text"])
    tfidf_vector = tfidf.transform(training_data["text"])
    model = kNN(n_neighbors = 7, weights='distance')
    model.fit(tfidf_vector, training_data["sentiment"])
    test(model, testing_data, tfidf)
    f = open("model.knn","wb")
    pickle.dump(model, f)
    f.close()
    f = open("corpus.tfidf","wb")
    pickle.dump(tfidf, f)
    f.close()
    

def test(model, data, tfidf):
    truth = numpy.array(data.sentiment)
    tfidf_vector = tfidf.transform(data["text"])
    result = model.predict(tfidf_vector)
    print(metrics.classification_report(truth, result, target_names=['negative', 'neutral', 'positive']))

train("cleaned_data.csv")
