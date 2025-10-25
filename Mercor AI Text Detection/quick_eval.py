import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

train=pd.read_csv('data/train.csv')
y=train['is_cheating']

vectorizers=[
    ('char_3_7', TfidfVectorizer(analyzer='char', ngram_range=(3,7), min_df=2, max_df=0.99)),
    ('charwb_2_6', TfidfVectorizer(analyzer='char_wb', ngram_range=(2,6), min_df=2, max_df=0.99)),
    ('word_1_3', TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=2, max_df=0.9, sublinear_tf=True)),
]

skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, vec in vectorizers:
    pipeline=Pipeline([
        ('tfidf', vec),
        ('clf', LogisticRegression(max_iter=5000, C=2.5, class_weight='balanced'))
    ])
    scores=cross_val_score(pipeline, train['answer'], y, cv=skf, scoring='roc_auc')
    print(name, scores, scores.mean())
