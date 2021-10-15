"""
Loads the saved baseline model and evaluates on testset. 

'train_baseline_crf.py' should be run first.
"""

import pickle
import pandas as pd
from baseline.src.utils import *
import nltk
from nltk.tokenize import WordPunctTokenizer
# from sklearn_crfsuite import metrics
import sklearn
from collections import Counter

# output location
output_dir = "test_output.csv"

# import model

saved_model_dir = "baseline/saved_models/baseline_crf.sav"
try:
    crf = pickle.load(open(saved_model_dir, 'rb'))
except FileNotFoundError:
    print("No model .sav file found. Did you run train_baseline_crf.py first?")

# import data

df_testset_dir = ("preprocessed_data/df_testset.csv")
df_testset = pd.read_csv(df_testset_dir)

getter_testset = SentenceGetter(df_testset)
sentences_testset = getter_testset.sentences

X_test = [sent2features(s) for s in sentences_testset]
y_test = [sent2labels(s) for s in sentences_testset]

classes = list(
    {item for sublist in y_test for item in sublist}
        )

new_classes = list(set(classes) - {'O'}) # NONE class removed from evaluation.
new_classes = sorted(new_classes, key = lambda name: (name[1:], name[0])) # sort for clean results table

# PREDICTION AND EVALUATION

y_pred = crf.predict(X_test)

# flatten data

sentences_testset = [x[0] for x in flatten(sentences_testset)]
y_test_flattened = flatten(y_test)
y_pred_flattened = flatten(y_pred)

# generate csv

output_df = pd.DataFrame(
    {
        "text": sentences_testset,
        "true": y_test_flattened,
        "pred": y_pred_flattened
    }
)

output_df.to_csv(output_dir, index=False)