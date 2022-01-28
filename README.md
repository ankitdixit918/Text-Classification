# Text-Classification
Classify text by leveraging NLP techniques to pre-process data before running ML classifiers such as Logistic Regression, Random Forest & MultinomialNB for the purpose of classification. During the model selection phase, Logistic Regression was chosen as the final model for building the classifier.

The files mentioned below are used in this project along with a brief description of the roles these files play

1. The file, **Text-Classification.ipynb** consists of all the steps taken in order to build a text classifier which includes text pre-processing & model selection
2. The script, **model_builder.py** is used to build a model & store it by pickling 
3. The script, **classifier.py** consists of the class, **TextClassifier** which has all the necessary methods & attributes in order to classify the text by leveraging the pickled model from previous step
4. The file, **app.py** has the necessary code which acts as the API for any requests that are routed in order to predict the class/category of a text
