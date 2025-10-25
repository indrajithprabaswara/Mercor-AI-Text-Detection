import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train=pd.read_csv('data/train.csv')
y=train['is_cheating'].values

vec=TfidfVectorizer(analyzer='char', ngram_range=(3,7), min_df=2, max_df=0.99)
X=vec.fit_transform(train['answer'])

skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof=np.zeros(len(train))

for fold,(tr_idx, val_idx) in enumerate(skf.split(X, y)):
    clf=LogisticRegression(max_iter=5000, C=2.5, class_weight='balanced', solver='lbfgs')
    clf.fit(X[tr_idx], y[tr_idx])
    oof[val_idx]=clf.predict_proba(X[val_idx])[:,1]
    auc=roc_auc_score(y[val_idx], oof[val_idx])
    print('fold', fold+1, 'auc', auc)

auc=roc_auc_score(y, oof)
print('overall oof auc', auc)

pred_labels=(oof>0.5).astype(int)
train['oof_prob']=oof
train['oof_pred']=pred_labels
train['correct']=(pred_labels==y)
print('accuracy', train['correct'].mean())
print(train[~train['correct']][['id','topic','answer','is_cheating','oof_prob']])
