"""
Loads the saved baseline model and asks for user prompts to predict. 

'train_baseline_crf.py' should be run first.
"""

import pickle
import pandas as pd
from baseline.src.utils import *
import nltk
from nltk.tokenize import WordPunctTokenizer

saved_model_dir = "baseline/saved_models/baseline_crf.sav"
try:
    crf = pickle.load(open(saved_model_dir, 'rb'))
except FileNotFoundError:
    print("No model .sav file found. Did you run train_baseline_crf.py first?")

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

    for word in WordPunctTokenizer().tokenize(text):
        if word: # i.e. not line break
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