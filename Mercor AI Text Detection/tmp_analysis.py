import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

train=pd.read_csv('data/train.csv')

def feats(df):
    s=df['answer'].str.lower()
    topic=df['topic'].str.lower()
    out=pd.DataFrame({
        'ans_char': df['answer'].str.len(),
        'ans_word': df['answer'].str.split().apply(len),
        'ans_unique': df['answer'].str.split().apply(lambda x: len(set(x)) if isinstance(x, list) else 0),
        'ans_upper_ratio': df['answer'].str.count(r'[A-Z]')/(df['answer'].str.len()+1),
        'ans_avg_word': df['answer'].str.len()/(df['answer'].str.split().apply(len)+1),
        'ans_digit_ratio': df['answer'].str.count(r'[0-9]')/(df['answer'].str.len()+1),
        'ans_punct_ratio': df['answer'].str.count(r'[.,;:!?]')/(df['answer'].str.len()+1),
        'ans_quote_ratio': df['answer'].str.count('"')/(df['answer'].str.len()+1),
        'ans_stop_count': s.str.count(' the ')+s.str.startswith('the ').astype(int),
        'ans_ai': s.str.contains(' ai').astype(int),
        'ans_llm': s.str.contains('chatgpt|gpt|large language model').astype(int),
        'topic_char': df['topic'].str.len(),
        'topic_word': df['topic'].str.split().apply(len),
        'topic_quote': df['topic'].str.contains('"').astype(int)
    })
    out['ans_unique_ratio']=out['ans_unique']/(out['ans_word']+1)
    out['topic_char_per_word']=out['topic_char']/ (out['topic_word']+1)
    return out.fillna(0)

X=feats(train)
y=train['is_cheating']
clf=GradientBoostingClassifier(random_state=42)
print('AUC', cross_val_score(clf,X,y,cv=5, scoring='roc_auc'))
print('ACC', cross_val_score(clf,X,y,cv=5, scoring='accuracy'))
