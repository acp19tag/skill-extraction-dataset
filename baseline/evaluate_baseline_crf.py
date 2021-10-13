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

y_test_flattened = flatten(y_test)
y_pred_flattened = flatten(y_pred)

print(
    sklearn.metrics.classification_report(
        y_test_flattened, y_pred_flattened, labels=new_classes
    )
)

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
        
print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(20))
print()

print("\nTop unlikely transitions:")
print_transitions(Counter(crf.transition_features_).most_common()[-20:])
print()

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))
        
print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))
print()

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])
print()