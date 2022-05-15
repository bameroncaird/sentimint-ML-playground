# Import SentiMint_kNN, do NOT import any other kNN files
import SentiMint_kNN

# To use the algorithm you must first create an instance of sentiMint_kNN
kNN = SentiMint_kNN.sentiMint_kNN()

# use .predict() to get a score {-100, 0, 100} on a string
# pass a SINGLE STRING. it will return a number
print(str(kNN.predict("Good day")))
