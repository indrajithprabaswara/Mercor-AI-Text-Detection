import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

train=pd.read_csv('data/train.csv')
X=train['answer']
y=train['is_cheating']

vect=TfidfVectorizer(analyzer='char', ngram_range=(3,6), min_df=2)
Xv=vect.fit_transform(X)
knn=KNeighborsClassifier(n_neighbors=1, metric='cosine')
print(cross_val_score(knn,Xv,y,cv=5,scoring='accuracy'))
