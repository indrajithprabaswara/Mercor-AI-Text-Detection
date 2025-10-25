import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

train=pd.read_csv('data/train.csv')
y=train['is_cheating']

vec=TfidfVectorizer(analyzer='char', ngram_range=(2,7), min_df=2, max_df=0.995)
X=vec.fit_transform(train['answer'])
clf=SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=5e-5, l1_ratio=0.25, max_iter=2000, class_weight='balanced', random_state=42)

skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores=[]
for train_idx, test_idx in skf.split(X, y):
    clf.fit(X[train_idx], y.iloc[train_idx])
    proba=clf.predict_proba(X[test_idx])[:,1]
    scores.append(roc_auc_score(y.iloc[test_idx], proba))
print(scores, np.mean(scores))
