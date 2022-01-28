from classifier import TextClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from classifier import TextClassifier
import langdetect
from langdetect import detect
from copy import deepcopy
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import  word_tokenize
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle

def build_model():
    model = TextClassifier()

    #with open('C:\\Users\\ankit\\Documents\\Text_Classification\\Context.csv') as f:
    data = pd.read_csv('C:\\Users\\ankit\\Documents\\Text_Classification\\Context.csv')

    df_txt_processing = deepcopy(data)

    #Function to detect language of text
    def func_detect(x):
        try:
            return detect(x)
        except:
            pass
    df_txt_processing['lang_1'] =  df_txt_processing['Text'].apply(lambda x: func_detect(x))

    #Drop rows which are not in english language
    df_txt_processing = df_txt_processing[df_txt_processing['lang_1'] == "en"]

    #Convert to lowercase
    df_txt_processing['Text'] = df_txt_processing['Text'].apply( lambda x: " ".join([i for i in x.lower().split()]) )

    #Lemmatization
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    df_txt_processing['Text'] = df_txt_processing['Text'].apply( lambda x: " ".join(lemma.lemmatize(word) for word in x.split()) )

    #Punctuation removal
    #df_txt_processing['Text'] = df_txt_processing['Text'].apply( lambda x: ''.join(ch for ch in x if ch not in exclude) )

    #Label encode the target variable
    df_txt_processing['Context_ID'] = df_txt_processing['Context/Topic'].factorize()[0]

    #Transform the words into vectors by leveraging TfidfVectorizer

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3, norm='l2', ngram_range=(1, 5), stop_words='english')
    features = tfidf.fit_transform(df_txt_processing.Text).toarray()
    labels = df_txt_processing.Context_ID
    features.shape

    context_ID_df = df_txt_processing[['Context/Topic', 'Context_ID']].drop_duplicates().sort_values('Context_ID')
    context_to_ID = dict(context_ID_df.values)
    ID_to_context = dict(context_ID_df[['Context_ID', 'Context/Topic']].values)

    #Look at the common unigrams, bigrams & trigrams
    N = 3
    for topic, context_ID in sorted(context_to_ID.items()):
        features_chi2 = chi2(features, labels == context_ID)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
        print("# '{}':".format(topic))
        print("Most correlated unigrams:\n* {}".format('\n* '.join(unigrams[-N:])))
        print("Most correlated bigrams:\n* {}".format('\n* '.join(bigrams[-N:])))
        print("Most correlated trigrams:\n* {}".format('\n* '.join(trigrams[-N:])))

    #Split the data into train & test dataset    
    train=df_txt_processing.sample(frac=0.8, random_state=200) #random state is a seed value
    test=df_txt_processing.drop(train.index)

    #Create feature vectors from textual data
    model.vectorizer_fit(train['Text'])
    print('Vectorizer fit complete')
    X = model.vectorizer_transform(train['Text'])
    print('Vectorizer transform complete')
    y = train['Context/Topic']

    #Train the model
    model.train(X, y)
    #Export the model & feature vectors
    model.pickle_clf()
    model.pickle_vectorizer()


if __name__ == "__main__":
    build_model()
