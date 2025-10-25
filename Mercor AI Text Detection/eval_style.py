import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

train=pd.read_csv('data/train.csv')

def build_features(df):
    text=df['answer'].fillna('')
    topic=df['topic'].fillna('')
    words=text.str.split()
    word_counts=words.apply(len)
    char_counts=text.str.len()
    unique_counts=words.apply(lambda w: len(set(w)) if w else 0)
    sentence_counts=text.str.count(r'[.!?]')+1
    comma_counts=text.str.count(',')
    period_counts=text.str.count(r'\.')
    question_counts=text.str.count('\?')
    exclamation_counts=text.str.count('!')
    digit_counts=text.str.count(r'[0-9]')
    uppercase_counts=text.str.count(r'[A-Z]')
    newline_counts=text.str.count('\n')
    connectors=['in conclusion','in summary','furthermore','moreover','additionally','however','therefore','thus','consequently','as a result','on the other hand','for instance','for example','it is important to note','it is worth noting','that being said']
    formal=['utilize','facilitate','implement','methodology','paradigm','leverage','robust','optimal','enhance','demonstrate']
    passive=['is made','was made','is given','was given','is shown','was shown']
    data=pd.DataFrame({
        'char_count':char_counts,
        'word_count':word_counts,
        'unique_words':unique_counts,
        'avg_word_len':char_counts/(word_counts+1),
        'sentence_count':sentence_counts,
        'avg_sentence_len':word_counts/(sentence_counts+1),
        'comma_count':comma_counts,
        'period_count':period_counts,
        'question_count':question_counts,
        'exclamation_count':exclamation_counts,
        'digit_count':digit_counts,
        'uppercase_count':uppercase_counts,
        'newline_count':newline_counts,
        'connector_hits':text.apply(lambda x: sum(p in x.lower() for p in connectors)),
        'formal_hits':text.apply(lambda x: sum(p in x.lower() for p in formal)),
        'passive_hits':text.apply(lambda x: sum(p in x.lower() for p in passive)),
        'quote_count':text.str.count('"'),
        'colon_count':text.str.count(':'),
        'semicolon_count':text.str.count(';'),
        'topic_char':topic.str.len(),
        'topic_word':topic.str.split().apply(len),
        'topic_quoted':topic.str.contains('"').astype(int),
        'the_count':text.str.count(r'\bthe\b'),
        'and_count':text.str.count(r'\band\b'),
        'capital_ratio':uppercase_counts/(char_counts+1),
        'digit_ratio':digit_counts/(char_counts+1),
        'punct_ratio':(comma_counts+period_counts+question_counts+exclamation_counts)/(char_counts+1),
        'unique_ratio':unique_counts/(word_counts+1)
    })
    return data.fillna(0)

X=build_features(train)
y=train['is_cheating']

skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf=RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6, min_samples_leaf=3)
print('RF', cross_val_score(rf, X, y, cv=skf, scoring='roc_auc'))
gb=GradientBoostingClassifier(random_state=42, n_estimators=500, learning_rate=0.02, max_depth=3)
print('GB', cross_val_score(gb, X, y, cv=skf, scoring='roc_auc'))
