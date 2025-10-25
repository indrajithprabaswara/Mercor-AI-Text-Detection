import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

train=pd.read_csv('data/train.csv')
X=train['answer']
y=train['is_cheating']

vect=TfidfVectorizer(analyzer='char', ngram_range=(3,5))
Xv=vect.fit_transform(X)
clf=LinearSVC(C=1.0,class_weight='balanced')
print(cross_val_score(clf,Xv,y,cv=5, scoring='accuracy'))
