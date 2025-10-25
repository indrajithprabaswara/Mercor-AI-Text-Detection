import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

train=pd.read_csv('data/train.csv')
X=train['answer']
y=train['is_cheating']

pipeline=Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3,6), min_df=2)),
    ('clf', ComplementNB())
])
print(cross_val_score(pipeline,X,y,cv=5,scoring='roc_auc'))
