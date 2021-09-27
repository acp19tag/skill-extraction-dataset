# Import packages

import os
from re import VERBOSE
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, make_scorer
import scipy.stats
from baseline.src.utils import *
# import sklearn_crfsuite
from sklearn_crfsuite import CRF, scorers, metrics
from collections import Counter

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# import preprocessing scripts from parent directory

# sys.path.insert(1, os.path.join(sys.path[0], '../..'))
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
For using established train/test sets
"""
X_train = [sent2features(s) for s in sentences_answers]
y_train = [sent2labels(s) for s in sentences_answers]

X_test = [sent2features(s) for s in sentences_testset]
y_test = [sent2labels(s) for s in sentences_testset]

"""
For splitting testset into test/train
"""
# X = [sent2features(s) for s in sentences_answers]
# y = [sent2labels(s) for s in sentences_answers]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size = 0.33, random_state = 0 
# )


classes = list(
    {item for sublist in y_train for item in sublist}.union(
        {item for sublist in y_train for item in sublist}
        ))

new_classes = list(set(classes) - {'O'}) # NONE class removed from evaluation.
new_classes = sorted(new_classes, key = lambda name: (name[1:], name[0])) # sort for clean results table

print('*** Data statistics: ***\n')
print(f'Length of training set: \t{len(X_train)} sentences\n')
print(f'Length of test set: \t{len(X_test)} sentences\n')
print('Classes present in training data:\n')
for class_label in sorted(classes):
    print(f'- {class_label}')
print()
print('Training baseline model...')

# # ORIGINAL CRF HERE
# crf = CRF(
#     algorithm = 'lbfgs',
#     c1 = 0.1,
#     c2 = 0.1,
#     max_iterations = 100,
#     all_possible_transitions = True
# )

# crf.fit(X_train, y_train)

crf = CRF(
    algorithm = 'lbfgs',
    max_iterations = 100,
    all_possible_transitions = True
)

params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05)
}

f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels = classes)

rs = RandomizedSearchCV(crf, params_space,
                            cv = 3, 
                            verbose = 1, 
                            n_jobs = -1, 
                            n_iter = 50,
                            scoring = f1_scorer)

rs.fit(X_train, y_train)

crf = rs.best_estimator_

print('best parameters:', rs.best_params_)
print('best CV score:', rs.best_score_)
print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

_x = [s.parameters['c1'] for s in rs.grid_scores_]
_y = [s.parameters['c2'] for s in rs.grid_scores_]
_c = [s.mean_validation_score for s in rs.grid_scores_]

fig = plt.figure()
fig.set_size_inches(12, 12)
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
    min(_c), max(_c)
))

ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])

print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))


# PREDICTION AND EVALUATION

y_pred = crf.predict(X_test)

print(
    metrics.flat_classification_report(
        y_test, y_pred, labels = new_classes
    )
)

# print()

# def print_transitions(trans_features):
#     for (label_from, label_to), weight in trans_features:
#         print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
        
# print("Top likely transitions:")
# print_transitions(Counter(crf.transition_features_).most_common(20))
# print()

# print("\nTop unlikely transitions:")
# print_transitions(Counter(crf.transition_features_).most_common()[-20:])
# print()

# def print_state_features(state_features):
#     for (attr, label), weight in state_features:
#         print("%0.6f %-8s %s" % (weight, label, attr))
        
# print("Top positive:")
# print_state_features(Counter(crf.state_features_).most_common(30))
# print()

# print("\nTop negative:")
# print_state_features(Counter(crf.state_features_).most_common()[-30:])
# print()