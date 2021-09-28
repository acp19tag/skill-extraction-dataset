"""
Loads the saved baseline model and asks for user prompts to predict. 

'train_baseline_crf.py' should be run first.
"""

from sklearn_crfsuite import CRF, metrics
from sklearn.metrics import classification_report
import pickle
import pandas as pd
from baseline.src.utils import *
import nltk
from nltk.tokenize import WordPunctTokenizer

saved_model_dir = "baseline/saved_models/baseline_crf.sav"
crf = pickle.load(open(saved_model_dir, 'rb'))

# df_testset_dir = ("preprocessed_data/df_testset.csv")
# df_testset = pd.read_csv(df_testset_dir)

# getter_testset = SentenceGetter(df_testset)
# sentences_testset = getter_testset.sentences

# X_test = [sent2features(s) for s in sentences_testset]
# y_test = [sent2labels(s) for s in sentences_testset]

# classes = list(
#     {item for sublist in y_test for item in sublist} - {'O'}
#     ) 

# y_pred = crf.predict(X_test)

# print(
#     metrics.flat_classification_report(
#         y_test, y_pred, labels = classes
#     )
# )

def convert_prompt_to_df(text):
    """ 
    converts the text to SentenceGetter-parseable dataframe.
    See txt_to_df in preprocessing/src/utils.py for more info. 
    """

    sentence_number = 1
    sentence_number_list = []
    word_list = []
    pos_list = []
    tag_list = [] # this keeps the architecture intact but will not be used
    rolling_sentence_list = []
    previous_line_break = True

    num_lines = sum(1 for line in text.splitlines())

    for word in WordPunctTokenizer().tokenize(text):
        if word: # i.e. not line break
            # line_items = line.split()
            # word = line_items[0]
            # current_tag_list = line_items[1:]
            sentence_number_list.append(str(sentence_number))
            word_list.append(word)
            tag_list.append('?')
            rolling_sentence_list.append(word)

            previous_line_break = False

        elif not previous_line_break:
            previous_line_break = True
            for tag in [x[1] for x in nltk.pos_tag(rolling_sentence_list)]:
                pos_list.append(tag)
            rolling_sentence_list = []
            sentence_number += 1
    # and again for the last sentence..
    for tag in [x[1] for x in nltk.pos_tag(rolling_sentence_list)]:
        pos_list.append(tag)

    return pd.DataFrame(
        {
            'sentence_id': sentence_number_list,
            'word': word_list, 
            'pos': pos_list,
            'tag': tag_list
        }
    )

def prepare_prompt(text):
    text_df = convert_prompt_to_df(text)
    getter_prompt = SentenceGetter(text_df)
    sentences_prompt = getter_prompt.sentences
    return [sent2features(s) for s in sentences_prompt]

while True:
    print()
    text = input("Enter text to be parsed (or 'quit' to cancel): ")
    if text in ['quit', 'exit', 'cancel', 'abort']:
        break
    X_test = prepare_prompt(text)
    y_pred = crf.predict(X_test)

    print()
    print("Predicted classes:")
    print()

    tokenized_text = WordPunctTokenizer().tokenize(text)

    pad_length = max(len(word) for word in tokenized_text)

    for word_tag_pair in zip(tokenized_text, y_pred[0]):
        print("{: <20} {: <20}".format(*word_tag_pair))

    print()