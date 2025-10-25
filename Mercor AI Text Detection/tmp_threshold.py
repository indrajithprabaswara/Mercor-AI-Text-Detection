import pandas as pd
import numpy as np
train=pd.read_csv('data/train.csv')
char_count=train['answer'].str.len()
word_count=train['answer'].str.split().apply(len)
avg_word=char_count/(word_count+1)
train['avg_word']=avg_word
best=(0,None)
for thresh in np.linspace(0,20,400):
    preds=(avg_word>thresh).astype(int)
    acc=(preds==train['is_cheating']).mean()
    if acc>best[0]:
        best=(acc,thresh)
print(best)
