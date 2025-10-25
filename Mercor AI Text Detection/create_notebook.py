import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("# Mercor AI Text Detection - Stacked Ensemble"))
cells.append(nbf.v4.new_markdown_cell(
"This notebook builds a stacked ensemble that blends stylometric features with several TF-IDF based linear models. "
"The goal is to produce highly confident predictions (targeting perfect accuracy) for the Mercor AI Text Detection Kaggle competition."
))

imports_code = """
import numpy as np
import pandas as pd
from pathlib import Path

from IPython.display import display
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
pd.set_option('display.max_colwidth', 120)
""".strip()

cells.append(nbf.v4.new_code_cell(imports_code))

load_code = """
DATA_DIR = Path('data')
train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'test.csv')
sample_submission = pd.read_csv(DATA_DIR / 'sample_submission.csv')

y = train_df['is_cheating'].values

print(f'Train shape: {train_df.shape}')
print(f'Test shape: {test_df.shape}')
print('\nTraining label distribution (counts / %):')
display(train_df['is_cheating'].value_counts().to_frame('count').assign(percent=lambda df: df['count'] / df['count'].sum() * 100))
""".strip()

cells.append(nbf.v4.new_code_cell(load_code))

eda_code = """
train_char_stats = train_df['answer'].str.len().describe()
train_word_stats = train_df['answer'].str.split().apply(len).describe()
print('Character count stats for answers:')
display(train_char_stats.to_frame().T)
print('Word count stats for answers:')
display(train_word_stats.to_frame().T)
""".strip()

cells.append(nbf.v4.new_code_cell(eda_code))

cells.append(nbf.v4.new_markdown_cell('## Stylometric feature builder'))

stylometric_code = """
def build_stylometric_features(df: pd.DataFrame) -> np.ndarray:
    # Compute dense stylometric statistics for answer/topic text.
    data = df.copy()
    text = data['answer'].fillna('')
    topic = data['topic'].fillna('')

    words = text.str.split()
    word_counts = words.apply(len)
    unique_counts = words.apply(lambda tokens: len(set(tokens)) if tokens else 0)
    char_counts = text.str.len()

    vowel_counts = text.str.count(r'[aeiouAEIOU]')
    digit_counts = text.str.count(r'[0-9]')
    punctuation_counts = text.str.count(r'[.,;:!?]')
    uppercase_counts = text.str.count(r'[A-Z]')
    sentence_counts = text.str.count(r'[.!?]') + 1
    newline_counts = text.str.count('\\n')

    avg_word_len = char_counts / (word_counts + 1)
    unique_ratio = unique_counts / (word_counts + 1)
    vowel_ratio = vowel_counts / (char_counts + 1)
    digit_ratio = digit_counts / (char_counts + 1)
    punctuation_ratio = punctuation_counts / (char_counts + 1)
    uppercase_ratio = uppercase_counts / (char_counts + 1)
    words_per_sentence = word_counts / sentence_counts

    topic_lengths = topic.str.len()
    topic_word_counts = topic.str.split().apply(len)
    topic_avg_word_len = topic_lengths / (topic_word_counts + 1)

    features = np.vstack([
        char_counts,
        word_counts,
        unique_counts,
        avg_word_len,
        unique_ratio,
        vowel_ratio,
        digit_ratio,
        punctuation_ratio,
        uppercase_ratio,
        words_per_sentence,
        topic_lengths,
        topic_word_counts,
        topic_avg_word_len,
        newline_counts,
    ]).astype(np.float32).T

    return features
""".strip()

cells.append(nbf.v4.new_code_cell(stylometric_code))

cells.append(nbf.v4.new_markdown_cell('## Base learner configuration'))

base_models_code = """
BASE_MODELS = [
    (
        'sgd_char_3_6',
        Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3, 6), min_df=2, max_df=0.95)),
            ('clf', SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.15, alpha=1e-4,
                                  max_iter=4000, class_weight='balanced', n_iter_no_change=20,
                                  random_state=RANDOM_STATE)),
        ]),
        'answer'
    ),
    (
        'sgd_charwb_3_5',
        Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_df=0.95)),
            ('clf', SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.2, alpha=5e-5,
                                  max_iter=4000, class_weight='balanced', n_iter_no_change=20,
                                  random_state=RANDOM_STATE)),
        ]),
        'answer'
    ),
    (
        'sgd_word_1_3',
        Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, max_df=0.95, sublinear_tf=True)),
            ('clf', SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.15, alpha=5e-4,
                                  max_iter=3000, class_weight='balanced', n_iter_no_change=20,
                                  random_state=RANDOM_STATE)),
        ]),
        'answer'
    ),
    (
        'logreg_word_1_2',
        Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_df=0.9)),
            ('clf', LogisticRegression(max_iter=3000, C=2.0, class_weight='balanced', solver='lbfgs',
                                      random_state=RANDOM_STATE)),
        ]),
        'answer'
    ),
    (
        'logreg_char_4_7',
        Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(4, 7), min_df=2, max_df=0.95)),
            ('clf', LogisticRegression(max_iter=4000, C=1.5, class_weight='balanced', solver='lbfgs',
                                      random_state=RANDOM_STATE)),
        ]),
        'answer'
    ),
    (
        'topic_logreg',
        Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, max_df=0.95)),
            ('clf', LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
        ]),
        'topic'
    ),
    (
        'style_logreg',
        Pipeline([
            ('features', FunctionTransformer(lambda X: build_stylometric_features(pd.DataFrame(X, columns=['answer', 'topic'])),
                                             validate=False)),
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', LogisticRegression(max_iter=500, C=2.0, class_weight='balanced', random_state=RANDOM_STATE)),
        ]),
        'both'
    ),
]
""".strip()

cells.append(nbf.v4.new_code_cell(base_models_code))

cells.append(nbf.v4.new_markdown_cell('## Cross-validated stacking (level-1 predictions)'))

stacking_code = """
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
n_models = len(BASE_MODELS)
oof_predictions = np.zeros((len(train_df), n_models), dtype=np.float32)

for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df, y), start=1):
    fold_train = train_df.iloc[train_idx]
    fold_valid = train_df.iloc[valid_idx]
    y_train = y[train_idx]

    for model_idx, (name, estimator, feature_key) in enumerate(BASE_MODELS):
        model = clone(estimator)

        if feature_key == 'answer':
            model.fit(fold_train['answer'], y_train)
            oof_predictions[valid_idx, model_idx] = model.predict_proba(fold_valid['answer'])[:, 1]
        elif feature_key == 'topic':
            model.fit(fold_train['topic'], y_train)
            oof_predictions[valid_idx, model_idx] = model.predict_proba(fold_valid['topic'])[:, 1]
        else:
            model.fit(fold_train[['answer', 'topic']], y_train)
            oof_predictions[valid_idx, model_idx] = model.predict_proba(fold_valid[['answer', 'topic']])[:, 1]

    print(f'Fold {fold} finished.')

base_scores = (
    pd.DataFrame(
        {
            'model': [name for name, _, _ in BASE_MODELS],
            'roc_auc': [roc_auc_score(y, oof_predictions[:, idx]) for idx in range(n_models)],
        }
    )
    .sort_values('roc_auc', ascending=False)
    .reset_index(drop=True)
)

display(base_scores)
""".strip()

cells.append(nbf.v4.new_code_cell(stacking_code))

cells.append(nbf.v4.new_markdown_cell('## Train meta-learner'))

meta_code = """
meta_model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
meta_model.fit(oof_predictions, y)
oof_meta = meta_model.predict_proba(oof_predictions)[:, 1]
meta_auc = roc_auc_score(y, oof_meta)
print(f'Meta-model ROC-AUC on out-of-fold predictions: {meta_auc:.6f}')
""".strip()

cells.append(nbf.v4.new_code_cell(meta_code))

cells.append(nbf.v4.new_markdown_cell('## Fit base learners on all data & export submission'))

final_code = """
full_test_predictions = np.zeros((len(test_df), len(BASE_MODELS)), dtype=np.float32)

for model_idx, (name, estimator, feature_key) in enumerate(BASE_MODELS):
    model = clone(estimator)

    if feature_key == 'answer':
        model.fit(train_df['answer'], y)
        full_test_predictions[:, model_idx] = model.predict_proba(test_df['answer'])[:, 1]
    elif feature_key == 'topic':
        model.fit(train_df['topic'], y)
        full_test_predictions[:, model_idx] = model.predict_proba(test_df['topic'])[:, 1]
    else:
        model.fit(train_df[['answer', 'topic']], y)
        full_test_predictions[:, model_idx] = model.predict_proba(test_df[['answer', 'topic']])[:, 1]

final_predictions = meta_model.predict_proba(full_test_predictions)[:, 1]
submission = pd.DataFrame({'id': test_df['id'], 'is_cheating': final_predictions})
submission_path = Path('submission.csv')
submission.to_csv(submission_path, index=False)
print(f'Submission file written to {submission_path.resolve()}')
display(submission.head())
""".strip()

cells.append(nbf.v4.new_code_cell(final_code))

nb['cells'] = cells

output_path = Path('mercor_ai_detection_solution.ipynb')
output_path.write_text(nbf.writes(nb), encoding='ascii')
print(f'Notebook written to {output_path.resolve()}')
