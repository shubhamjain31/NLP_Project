#importing libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#reading a file from dataframe
df = pd.read_csv('App/Files/Spam.csv')

#rename a columns[v1 and v2]
df.rename(columns={"v1":"labels","v2":"messages"},inplace=True)

#drop all columns with NaN values
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True) 

#convert the labels in 0 and 1 form
df['label'] = df['labels'].map({'ham':0,'spam':1})

#selecting dependent and independent variables
X = df['messages']
y = df['label']

#Training Naive bayes classifier
cv = CountVectorizer()
X = cv.fit_transform(X)

#Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

#selecting a model for training a dataset
clf = MultinomialNB()
clf.fit(X_train,y_train)

#score of model
clf.score(X_test,y_test)

#predicting the model
y_pred = clf.predict(X_test)

#function for analysis the text
def spam_analyse(spam_text):
	message = spam_text
	data = [message]
	vect = cv.transform(data).toarray()
	prediction = clf.predict(vect)
	return prediction
