import pandas
import numpy
import pickle


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# POSNEG dictionary used for easily translating the dataset labels to numbers 
# that will be returned and that the algorithm can better understand

# negative := -100
# neutral  :=  0
# positive :=  100
POSNEG = {"negative": -100, "neutral": 0, "positive": 100}

# Used to train a model and generate a library for the given data file
# File should be a csv 
def kNN_train(data_file_name):

    # get data from file and put it into data lists
    data = pandas.read_csv(data_file_name)[["text","sentiment"]]

    # Splitting data same way as LR and SVM for consistincy
    training_data, test_data = train_test_split(data, test_size=0.1)

    # This is just chaning the labels to numeric values
    training_data["sentiment"] = training_data["sentiment"].map(POSNEG)
    test_data["sentiment"] = test_data["sentiment"].map(POSNEG)

    # shout out to Cameron. This prevents crashes occasonally
    training_data.fillna('',inplace=True)
    test_data.fillna('',inplace=True)
    
    # Create the trainig vector
    tfidf_vectorizer = TfidfVectorizer()

    # generate corpus IDF portion
    tfidf_vectorizer.fit(training_data["text"])

    # generate TF portion for training
    tfidf_vector = tfidf_vectorizer.transform(training_data["text"])

    # k = 7 and distance mode found to yeild best results
    k_nearest = KNeighborsClassifier(n_neighbors = 7, weights='distance')

    # Training time
    k_nearest.fit(tfidf_vector, training_data["sentiment"])

    # test model
    kNN_test(k_nearest, test_data, tfidf_vectorizer)

    # save model and vectorizer
    f = open("model.knn","wb")
    pickle.dump(k_nearest, f)
    f.close()
    f = open("corpus.tfidf","wb")
    pickle.dump(tfidf_vectorizer, f)
    f.close()
    

def kNN_test(k_nearest, data, tfidf_vectorizer):
    # Get answer key for correct labels
    key = numpy.array(data.sentiment)

    # generate TF for test data
    tfidf_vector = tfidf_vectorizer.transform(data["text"])

    # get predictions on test set
    result = k_nearest.predict(tfidf_vector)

    # get predictions on test set
    print(metrics.classification_report(key, result, target_names=['negative', 'neutral', 'positive']))

# Running this file will create a model
kNN_train("../SVM/cleaned_data.csv")
