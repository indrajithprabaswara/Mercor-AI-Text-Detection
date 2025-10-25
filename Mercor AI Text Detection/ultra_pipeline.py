
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

RANDOM_STATE = 128
np.random.seed(RANDOM_STATE)

DATA_DIR = Path('data')
train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'test.csv')
y = train_df['is_cheating'].values

def build_stylometric(df: pd.DataFrame) -> pd.DataFrame:
    text = df['answer'].fillna('')
    topic = df['topic'].fillna('')

    words = text.str.split()
    word_counts = words.apply(len)
    unique_counts = words.apply(lambda tok: len(set(tok)) if tok else 0)
    char_counts = text.str.len()

    sentence_counts = text.str.count(r'[.!?]') + 1
    comma_counts = text.str.count(',')
    punctuation_counts = text.str.count(r'[.,;:!?]')
    uppercase_counts = text.str.count(r'[A-Z]')
    digit_counts = text.str.count(r'[0-9]')
    newline_counts = text.str.count('\n')
    connector_terms = [
        'in conclusion', 'furthermore', 'moreover', 'additionally', 'however',
        'therefore', 'thus', 'consequently', 'as a result', 'for example',
        'for instance', 'overall', 'in summary'
    ]
    formal_terms = [
        'utilize', 'methodology', 'paradigm', 'robust', 'demonstrate',
        'optimise', 'framework', 'leverage', 'strategic', 'comprehensive'
    ]
    passive_terms = ['is made', 'was made', 'is given', 'was given', 'is shown', 'was shown']
    df_feat = pd.DataFrame({
        'char_count': char_counts,
        'word_count': word_counts,
        'unique_words': unique_counts,
        'avg_word_len': char_counts / (word_counts + 1),
        'sentence_count': sentence_counts,
        'avg_sentence_len': word_counts / (sentence_counts + 1),
        'comma_count': comma_counts,
        'punctuation_count': punctuation_counts,
        'uppercase_count': uppercase_counts,
        'digit_count': digit_counts,
        'newline_count': newline_counts,
        'connector_hits': text.apply(lambda x: sum(term in x.lower() for term in connector_terms)),
        'formal_hits': text.apply(lambda x: sum(term in x.lower() for term in formal_terms)),
        'passive_hits': text.apply(lambda x: sum(term in x.lower() for term in passive_terms)),
        'quote_count': text.str.count('"'),
        'colon_count': text.str.count(':'),
        'semicolon_count': text.str.count(';'),
        'exclamation_count': text.str.count('!'),
        'question_count': text.str.count(r'\?'),
        'topic_char': topic.str.len(),
        'topic_words': topic.str.split().apply(len),
        'topic_caps': topic.str.count(r'[A-Z]'),
    })
    df_feat['unique_ratio'] = df_feat['unique_words'] / (df_feat['word_count'] + 1)
    df_feat['punct_ratio'] = df_feat['punctuation_count'] / (df_feat['char_count'] + 1)
    df_feat['upper_ratio'] = df_feat['uppercase_count'] / (df_feat['char_count'] + 1)
    df_feat['digit_ratio'] = df_feat['digit_count'] / (df_feat['char_count'] + 1)
    df_feat['topic_char_per_word'] = df_feat['topic_char'] / (df_feat['topic_words'] + 1)
    return df_feat.fillna(0)

stylometric_train = build_stylometric(train_df)
stylometric_test = build_stylometric(test_df)

BASE_MODELS = [
    {
        'name': 'lr_char_3_7',
        'type': 'tfidf',
        'params': {
            'vectorizer': TfidfVectorizer(analyzer='char', ngram_range=(3, 7), min_df=2, max_df=0.99),
            'estimator': LogisticRegression(max_iter=6000, C=2.5, class_weight='balanced', solver='lbfgs'),
        }
    },
    {
        'name': 'lr_charwb_2_6',
        'type': 'tfidf',
        'params': {
            'vectorizer': TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), min_df=2, max_df=0.99),
            'estimator': LogisticRegression(max_iter=6000, C=3.0, class_weight='balanced', solver='lbfgs'),
        }
    },
    {
        'name': 'sgd_char_2_7',
        'type': 'tfidf',
        'params': {
            'vectorizer': TfidfVectorizer(analyzer='char', ngram_range=(2, 7), min_df=2, max_df=0.995),
            'estimator': SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=5e-5, l1_ratio=0.25, max_iter=4000, tol=1e-3, class_weight='balanced', n_iter_no_change=20, random_state=RANDOM_STATE),
        }
    },
    {
        'name': 'lr_word_1_3',
        'type': 'tfidf',
        'params': {
            'vectorizer': TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, max_df=0.9, sublinear_tf=True),
            'estimator': LogisticRegression(max_iter=5000, C=2.0, class_weight='balanced', solver='lbfgs'),
        }
    },
    {
        'name': 'gb_stylo',
        'type': 'stylo',
        'params': {
            'scaler': StandardScaler(),
            'estimator': GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=600, learning_rate=0.02, max_depth=3, subsample=0.9, min_samples_leaf=3),
        }
    },
    {
        'name': 'rf_stylo',
        'type': 'stylo',
        'params': {
            'scaler': None,
            'estimator': RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_leaf=4, random_state=RANDOM_STATE, n_jobs=-1),
        }
    }
]

n_models = len(BASE_MODELS)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=RANDOM_STATE)
oof_matrix = np.zeros((len(train_df), n_models), dtype=np.float32)
test_matrix = np.zeros((len(test_df), n_models), dtype=np.float32)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df, y), start=1):
    y_tr = y[train_idx]
    answer_tr = train_df['answer'].iloc[train_idx]
    answer_val = train_df['answer'].iloc[valid_idx]

    for model_pos, spec in enumerate(BASE_MODELS):
        if spec['type'] == 'tfidf':
            vectorizer = clone(spec['params']['vectorizer'])
            estimator = clone(spec['params']['estimator'])

            X_tr = vectorizer.fit_transform(answer_tr)
            X_val = vectorizer.transform(answer_val)
            estimator.fit(X_tr, y_tr)
            oof_matrix[valid_idx, model_pos] = estimator.predict_proba(X_val)[:, 1]

            X_test = vectorizer.transform(test_df['answer'])
            test_matrix[:, model_pos] += estimator.predict_proba(X_test)[:, 1] / skf.n_splits

        elif spec['type'] == 'stylo':
            features_tr = stylometric_train.iloc[train_idx]
            features_val = stylometric_train.iloc[valid_idx]
            estimator = clone(spec['params']['estimator'])
            scaler = spec['params']['scaler']

            if scaler is not None:
                scaler = clone(scaler)
                X_tr = scaler.fit_transform(features_tr)
                X_val = scaler.transform(features_val)
                X_test = scaler.transform(stylometric_test)
            else:
                X_tr = features_tr.values
                X_val = features_val.values
                X_test = stylometric_test.values

            estimator.fit(X_tr, y_tr)
            oof_matrix[valid_idx, model_pos] = estimator.predict_proba(X_val)[:, 1]
            test_matrix[:, model_pos] += estimator.predict_proba(X_test)[:, 1] / skf.n_splits
        else:
            raise ValueError(f"Unknown model type: {spec['type']}")

        gc.collect()

    blend_auc = roc_auc_score(y[valid_idx], oof_matrix[valid_idx].mean(axis=1))
    print(f"Fold {fold} finished | blended fold AUC: {blend_auc:.6f}")
model_scores = {}
for idx, spec in enumerate(BASE_MODELS):
    score = roc_auc_score(y, oof_matrix[:, idx])
    model_scores[spec['name']] = score

print('\nBase model OOF AUCs:')
for name, score in model_scores.items():
    print(f"  {name}: {score:.6f}")
meta_model = LogisticRegression(max_iter=6000, solver='lbfgs', random_state=RANDOM_STATE)
meta_model.fit(oof_matrix, y)
meta_oof = meta_model.predict_proba(oof_matrix)[:, 1]
meta_auc = roc_auc_score(y, meta_oof)
print(f"\nMeta-model OOF AUC: {meta_auc:.6f}")

calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(meta_oof, y)
calibrated_oof = calibrator.transform(meta_oof)
calibrated_auc = roc_auc_score(y, calibrated_oof)
print(f"Calibrated OOF AUC: {calibrated_auc:.6f}")
def confidence_adjustment(raw_scores: np.ndarray, stylometric_df: pd.DataFrame) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float64)
    adjusted = scores.copy()

    mask_mid = (adjusted > 0.38) & (adjusted < 0.62)
    if mask_mid.any():
        mask_series = pd.Series(mask_mid, index=stylometric_df.index)
        subset = stylometric_df.loc[mask_series]
        delta = np.zeros(len(subset), dtype=np.float64)
        delta += 0.08 * (subset['avg_word_len'].values > 5.7)
        delta += 0.05 * (subset['unique_ratio'].values < 0.58)
        delta -= 0.05 * (subset['unique_ratio'].values > 0.70)
        delta += 0.04 * (subset['connector_hits'].values >= 1)
        adjusted[mask_mid] += delta

    heavy_formal = stylometric_df['formal_hits'].values >= 2
    adjusted[heavy_formal] = np.clip(adjusted[heavy_formal] + 0.06, 0, 1)

    overt_creative = (stylometric_df['unique_ratio'].values > 0.75) & (stylometric_df['avg_word_len'].values < 5.2)
    adjusted[overt_creative] = np.clip(adjusted[overt_creative] - 0.07, 0, 1)

    return np.clip(adjusted, 0.001, 0.999)
adjusted_train = confidence_adjustment(calibrated_oof, stylometric_train)
train_accuracy = np.mean((adjusted_train >= 0.5) == y)
train_margin = np.min(np.abs(adjusted_train - y))
print(f"Adjusted train accuracy: {train_accuracy:.6f}")
print(f"Minimum margin to true label: {train_margin:.6f}")
full_test_matrix = np.zeros((len(test_df), n_models), dtype=np.float32)

for model_pos, spec in enumerate(BASE_MODELS):
    if spec['type'] == 'tfidf':
        vectorizer = clone(spec['params']['vectorizer'])
        estimator = clone(spec['params']['estimator'])
        X_train_full = vectorizer.fit_transform(train_df['answer'])
        X_test_full = vectorizer.transform(test_df['answer'])
        estimator.fit(X_train_full, y)
        full_test_matrix[:, model_pos] = estimator.predict_proba(X_test_full)[:, 1]
    elif spec['type'] == 'stylo':
        estimator = clone(spec['params']['estimator'])
        scaler = spec['params']['scaler']
        if scaler is not None:
            scaler = clone(scaler)
            X_train_full = scaler.fit_transform(stylometric_train)
            X_test_full = scaler.transform(stylometric_test)
        else:
            X_train_full = stylometric_train.values
            X_test_full = stylometric_test.values
        estimator.fit(X_train_full, y)
        full_test_matrix[:, model_pos] = estimator.predict_proba(X_test_full)[:, 1]
    gc.collect()
meta_test = meta_model.predict_proba(full_test_matrix)[:, 1]
calibrated_test = calibrator.transform(meta_test)
adjusted_test = confidence_adjustment(calibrated_test, stylometric_test)

submission = pd.DataFrame({'id': test_df['id'], 'is_cheating': adjusted_test})
submission_path = Path('submission.csv')
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved to {submission_path.resolve()} with range [{submission['is_cheating'].min():.4f}, {submission['is_cheating'].max():.4f}]")

submission.describe().to_csv('submission_stats.csv')

hard_cases = pd.DataFrame({
    'id': train_df['id'],
    'topic': train_df['topic'],
    'target': y,
    'meta_oof': meta_oof,
    'calibrated': calibrated_oof,
    'adjusted': adjusted_train,
})
hard_cases['margin'] = np.abs(hard_cases['adjusted'] - hard_cases['target'])
hard_cases.sort_values('margin', inplace=True)
hard_cases.to_csv('training_diagnostics.csv', index=False)
print('Training diagnostics written to training_diagnostics.csv')
