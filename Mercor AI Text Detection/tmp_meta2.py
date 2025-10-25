import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

train=pd.read_csv('data/train.csv')
test=pd.read_csv('data/test.csv')
y=train['is_cheating'].values


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
    newline=text.str.count('\\n')
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

base_models=[
    ('sgd_char36', Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3,6), min_df=2, max_df=0.95)),
        ('clf', SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.15, alpha=1e-4, max_iter=4000, class_weight='balanced', n_iter_no_change=20))
    ]), 'answer'),
    ('sgd_charwb', Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_df=0.95)),
        ('clf', SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.2, alpha=5e-5, max_iter=4000, class_weight='balanced', n_iter_no_change=20))
    ]), 'answer'),
    ('sgd_word', Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=2, max_df=0.95, sublinear_tf=True)),
        ('clf', SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.15, alpha=5e-4, max_iter=3000, class_weight='balanced', n_iter_no_change=20))
    ]), 'answer'),
    ('logreg_word', Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2, max_df=0.9)),
        ('clf', LogisticRegression(max_iter=3000, C=2.0, class_weight='balanced', solver='lbfgs'))
    ]), 'answer'),
    ('logreg_char', Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(4,7), min_df=2, max_df=0.95)),
        ('clf', LogisticRegression(max_iter=4000, C=1.5, class_weight='balanced', solver='lbfgs'))
    ]), 'answer'),
    ('topic_logreg', Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=1, max_df=0.95)),
        ('clf', LogisticRegression(max_iter=2000, C=1.0))
    ]), 'topic'),
    ('style_logreg', Pipeline([
        ('features', FunctionTransformer(stylometric_features, validate=False)),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, C=2.0, class_weight='balanced'))
    ]), 'both'),
]

skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds=np.zeros((len(train), len(base_models)))
test_preds=np.zeros((len(test), len(base_models)))

for fold, (tr_idx, val_idx) in enumerate(skf.split(train, y)):
    y_tr=y[tr_idx]
    fold_train=train.iloc[tr_idx]
    fold_val=train.iloc[val_idx]
    for m_idx, (name, model, kind) in enumerate(base_models):
        mdl=clone(model)
        if kind=='answer':
            mdl.fit(fold_train['answer'], y_tr)
            oof_preds[val_idx, m_idx]=mdl.predict_proba(fold_val['answer'])[:,1]
            test_preds[:, m_idx]+=mdl.predict_proba(test['answer'])[:,1]/skf.n_splits
        elif kind=='topic':
            mdl.fit(fold_train['topic'], y_tr)
            oof_preds[val_idx, m_idx]=mdl.predict_proba(fold_val['topic'])[:,1]
            test_preds[:, m_idx]+=mdl.predict_proba(test['topic'])[:,1]/skf.n_splits
        else:
            mdl.fit(fold_train[['answer','topic']], y_tr)
            oof_preds[val_idx, m_idx]=mdl.predict_proba(fold_val[['answer','topic']])[:,1]
            test_preds[:, m_idx]+=mdl.predict_proba(test[['answer','topic']])[:,1]/skf.n_splits
    print(f'Fold {fold+1} done')

for i,(name,_,_) in enumerate(base_models):
    auc=roc_auc_score(y, oof_preds[:,i])
    print(name, auc)

meta=LogisticRegression(max_iter=5000)
meta.fit(oof_preds, y)
oof_meta=meta.predict_proba(oof_preds)[:,1]
meta_auc=roc_auc_score(y, oof_meta)
print('meta AUC', meta_auc)

final_preds=meta.predict_proba(test_preds)[:,1]
print('pred range', final_preds.min(), final_preds.max())

submission=pd.DataFrame({'id': test['id'], 'is_cheating': final_preds})
submission.to_csv('submission_meta.csv', index=False)
print('saved submission_meta.csv')
