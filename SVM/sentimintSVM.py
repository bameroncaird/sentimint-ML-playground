import sklearn
import pandas
import numpy
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# Aiming for usability but I am somewhat new to Python so this using a lot of sklearn

POSNEG = {"negative": -100, "neutral": 0, "positive": 100}

def SVM_train(data_file_name):
    # get data
    data = pandas.read_csv(data_file_name)[["text","sentiment"]]
        
    # Splitting data same way as LR for consistincy
    training_data, test_data = train_test_split(data, test_size=0.1)

    training_data["sentiment"] = training_data["sentiment"].map(POSNEG)
    test_data["sentiment"] = test_data["sentiment"].map(POSNEG)

    training_data.fillna('',inplace=True)
    test_data.fillna('',inplace=True)

    # Create the trainig vector
    tfidf_training_vectorizer = TfidfVectorizer()
    tfidf_training_vectorizer.fit(data["text"])
    tfidf_vector = tfidf_training_vectorizer.transform(training_data["text"])
    #tfidf_vector = tfidf_training_vectorizer.transform(training_data["text"])

    # May set break_ties to true if we have the computational power to do so
    # Avoiding linear SVM due to Linear kernels generally having similar performance to LR
    SVC = svm.NuSVC(kernel = "rbf", break_ties=False)

    # I believe this trains it
    SVC.fit(tfidf_vector, training_data["sentiment"])
    SVM_test(SVC, test_data)
    

def SVM_test(SVC, data):
    key = numpy.array(data.sentiment)
    tfidf_training_vectorizer = TfidfVectorizer()
    tfidf_vector = tfidf_training_vectorizer.transform(data["text"])
    result = SVC.predict(tfidf_vector)

    
    accuracy = numpy.mean(key == result)
    print(str(accuracy))
    print(metrics.classification_report(key, result, target_names=['negative', 'neutral', 'positive']))

SVM_train("cleaned_data.csv")