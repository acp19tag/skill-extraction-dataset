"""
Trains, saves, and evaluates the baseline CRF model 
on provided training and test data. 
"""
# Import packages

import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from baseline.src.utils import *
import pickle
from sklearn_crfsuite import CRF, metrics
from collections import Counter
import sklearn

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# import preprocessing scripts from parent directory
from preprocessing.extra_preprocessing import *

# import data

df_answers_dir = ("preprocessed_data/df_answers.csv")
df_testset_dir = ("preprocessed_data/df_testset.csv")

if not os.path.isfile(df_answers_dir): # if the preprocessed data files do not yet exist, run the aggregation script to create them
    os.system('python -m preprocessing.data_aggregation')

df_answers = pd.read_csv(df_answers_dir)
df_testset = pd.read_csv(df_testset_dir)

# EXTRA PREPROCESSING
df_answers = extra_preprocessing(df_answers)

getter_answers = SentenceGetter(df_answers)
sentences_answers = getter_answers.sentences

getter_testset = SentenceGetter(df_testset)
sentences_testset = getter_testset.sentences

"""
Using established train/test sets...
"""
X_train = [sent2features(s) for s in sentences_answers]
y_train = [sent2labels(s) for s in sentences_answers]

X_test = [sent2features(s) for s in sentences_testset]
y_test = [sent2labels(s) for s in sentences_testset]

"""
... or, instead, splitting testset into test/train
"""
# from sklearn.model_selection import train_test_split

# X = [sent2features(s) for s in sentences_answers]
# y = [sent2labels(s) for s in sentences_answers]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size = 0.33, random_state = 0 
# )

classes = list(
    {item for sublist in y_train for item in sublist}.union(
        {item for sublist in y_test for item in sublist}
        ))

new_classes = list(set(classes) - {'O'}) # NONE class removed from evaluation.
new_classes = sorted(new_classes, key = lambda name: (name[1:], name[0])) # sort for clean results table

print('*** Data statistics: ***\n')
print(f'Length of training set: \t{len(X_train)} sentences\n')
print(f'Length of test set: \t\t{len(X_test)} sentences\n')
print('Classes present in training data:\n')
for class_label in sorted(classes):
    print(f'- {class_label}')
print()
print('Training baseline model...')

# MODEL TRAINING

crf = CRF(
    algorithm = 'lbfgs',
    c1 = 0.7476855308167297,    # best c1 parameter from RandomizedSearch
    c2 = 0.02292393494508309,   # best c2 parameter from RandomizedSearch
    max_iterations = 100,
    all_possible_transitions = True
)

crf.fit(X_train, y_train)

# SAVE MODEL WEIGHTS

saved_model_dir = "baseline/saved_models/"
saved_model_filename = "baseline_crf.sav"
# if directory doesn't exist, create it
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
pickle.dump(crf, open(saved_model_dir+saved_model_filename, 'wb'))

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