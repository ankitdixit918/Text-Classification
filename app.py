from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from classifier import TextClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
api = Api(app)

model = TextClassifier()

clf_path = 'C:\\Users\\ankit\\Documents\\Text_Classification\\models\\LogisticRegression_final_prediction.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'C:\\Users\\ankit\\Documents\\Text_Classification\\models\\TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument("query")


class PredictClass(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        print(args)
        user_query = args["query"] 

        # vectorize the user's query and make a prediction 
        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)

        # round the predict proba value and set to new variable
        confidence = round(pred_proba[0], 3)

        # create JSON object
        output = {'prediction': prediction, 'confidence': confidence}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictClass, '/')


if __name__ == '__main__':
    app.run(debug=True)
