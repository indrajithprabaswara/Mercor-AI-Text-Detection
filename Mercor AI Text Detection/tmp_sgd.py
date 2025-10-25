import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

train=pd.read_csv('data/train.csv')
X=train['answer']
y=train['is_cheating']

vect=TfidfVectorizer(analyzer='char', ngram_range=(3,6), min_df=2)
Xv=vect.fit_transform(X)
clf=SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.15, alpha=1e-4, max_iter=4000, class_weight='balanced', n_iter_no_change=20)
scores=cross_val_score(clf,Xv,y,cv=5,scoring='roc_auc')
print(scores)
print(scores.mean())
