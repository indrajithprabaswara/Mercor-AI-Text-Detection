import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer

train=pd.read_csv('data/train.csv')

def stylometric_features(X):
    df=pd.DataFrame(X, columns=['answer','topic'])
    text=df['answer'].fillna('')
    topic=df['topic'].fillna('')
    words=text.str.split()
    word_counts=words.apply(len)
    unique_counts=words.apply(lambda x: len(set(x)) if len(x)>0 else 0)
    char_counts=text.str.len()
    vowel=text.str.count(r'[aeiouAEIOU]')
    digit=text.str.count(r'[0-9]')
    punct=text.str.count(r'[.,;:!?]')
    uppercase=text.str.count(r'[A-Z]')
    sentences=text.str.count(r'[.!?]')+1
    avg_word=char_counts/(word_counts+1)
    unique_ratio=unique_counts/(word_counts+1)
    vowel_ratio=vowel/(char_counts+1)
    digit_ratio=digit/(char_counts+1)
    punct_ratio=punct/(char_counts+1)
    uppercase_ratio=uppercase/(char_counts+1)
    sentence_len=word_counts/(sentences)
    topic_len=topic.str.len()
    topic_word=topic.str.split().apply(len)
    topic_avg=topic_len/(topic_word+1)
    newline=text.str.count('\n')
    return np.vstack([
        char_counts,
        word_counts,
        unique_counts,
        avg_word,
        unique_ratio,
        vowel_ratio,
        digit_ratio,
        punct_ratio,
        uppercase_ratio,
        sentence_len,
        topic_len,
        topic_word,
        topic_avg,
        newline,
    ]).astype(float).T

column_transformer=ColumnTransformer([
    ('answer_word', TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2, max_df=0.9), 'answer'),
    ('answer_char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=3), 'answer'),
    ('topic_tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1,2)), 'topic'),
    ('stylometric', Pipeline([
        ('extract', FunctionTransformer(stylometric_features, validate=False)),
        ('scaler', StandardScaler())
    ]), ['answer','topic'])
])

pipeline=Pipeline([
    ('features', column_transformer),
    ('clf', LogisticRegression(max_iter=5000, C=4.0, class_weight='balanced'))
])

X=train[['answer','topic']]
y=train['is_cheating']

scores=cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
print('AUC', scores)
print(scores.mean())
