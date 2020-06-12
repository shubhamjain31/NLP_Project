import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df = pd.read_csv('App/Files/Spam.csv')
df.rename(columns={"v1":"labels","v2":"messages"},inplace=True)
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
df['label'] = df['labels'].map({'ham':0,'spam':1})
X = df['messages']
y = df['label']
cv = CountVectorizer()
X = cv.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)

def spam_analyse(spam_text):
	message = spam_text
	data = [message]
	vect = cv.transform(data).toarray()
	prediction = clf.predict(vect)
	return prediction
