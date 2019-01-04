#Sentiment analysis
import pandas as pd
import numpy as np

#importing the dataset
dataset = pd.read_csv('Training_Data.tsv', delimiter = '\t' , quoting = 3)
UnlabeledData = pd.read_csv('Unlabeled_Data.tsv' , delimiter = '\t' , quoting = 3 )

#cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
cleandata = []
for i in range(0, 2300):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
for i in range(0, 932):
    Treview = re.sub('[^a-zA-Z]', ' ', UnlabeledData['Text'][i])
    Treview = Treview.lower()
    Treview = Treview.split()
    ps = PorterStemmer()
    Treview = [ps.stem(word) for word in Treview if not word in set(stopwords.words('english'))]
    Treview = ' '.join(Treview)
    cleandata.append(Treview)

#Creating The Bag OF Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(corpus).toarray()
Z = cv.fit_transform(cleandata).toarray()
y = dataset.iloc[:, 1].values
'''Final_Z = np.pad(Z,((0,0),(0,1936)) , mode = 'constant')'''

#splitting the data-set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

#Fitting classifier To the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)
z_pred = classifier.predict(Z)


#Making The Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


