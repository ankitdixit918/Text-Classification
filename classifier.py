import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class TextClassifier(object):
    
    def __init__(self):
        """Simple NLP
        Attributes:
            clf: sklearn classifier model
            vectorizor: TFIDF vectorizer or similar
        """
        self.clf = LogisticRegression(multi_class='multinomial', random_state = 200,class_weight = 'balanced', max_iter=10000)
        
        self.vectorizer = TfidfVectorizer()
        
        #self.vect_word = TfidfVectorizer(max_features=20000, lowercase=True, analyzer='word',
        #                stop_words= 'english',ngram_range=(1,3),dtype=np.float32)
        #
        #self.vect_char = TfidfVectorizer(max_features=40000, lowercase=True, analyzer='char',
        #                stop_words= 'english',ngram_range=(3,6),dtype=np.float32)
        
    def vectorizer_fit(self, X):
        """Fits a TFIDF vectorizer to the text
        """
        
        self.vectorizer.fit(X)

    def vectorizer_transform(self, X):
        """Transform the text data to a sparse TFIDF matrix
        """
        
        X_transformed = self.vectorizer.transform(X) 
        return X_transformed
    

    def train(self, X, y):
        """Trains the classifier to associate the label with the sparse matrix
        """
        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """Returns probability for the binary class '1' in a numpy array
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    def pickle_vectorizer(self, path='C:\\Users\\ankit\\Documents\\Text_Classification\\models\\TFIDFVectorizer.pkl'):
        """Saves the trained vectorizer for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer at {}".format(path))

    def pickle_clf(self, path = 'C:\\Users\\ankit\\Documents\\Text_Classification\\models\\LogisticRegression_final_prediction.pkl'):
        """Saves the trained classifier for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))
