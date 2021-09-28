#! /usr/bin/python3
"""
contains extra preprocessing steps for raw data, including:
    - using regular expression to capture misclassified Skills in Experience class
    - separating terms with special characters (e.g. '/', ',')
"""

from preprocessing.src.utils import *   # pylint: disable=all
import re
import inflect                          # pylint: disable=all
import pandas as pd                     # pylint: disable=all
from pandas.core.common import SettingWithCopyWarning

# import warnings filter
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=SettingWithCopyWarning)

def get_class_from_tag(full_tag):
    """ strips the BIO prefix from the tag and returns the class """
    if full_tag == 'O':
        return full_tag
    return full_tag.split('-')[1]

def get_BIO_from_tag(full_tag):
    """ strips the class from the tag and returns the BIO prefix """
    if full_tag == 'O':
        return full_tag
    return full_tag.split('-')[0]

def identify_misclassified_exp(text):
    """ identifies whether a span classed as Exp is likely to be a misclassified Skill """

    misclassified = True
    
    # check if there is a valid number in number format (regex)
    if bool(re.search('[0-9]', text)):
        misclassified = False

    # check if there is a valid number in text format (inflect)
    inflect_engine = inflect.engine()
    text_numbers = {inflect_engine.number_to_words(x) for x in range(100)}

    for token in re.findall(r"[\w]+|[^\s\w]", text):
        if token.lower() in text_numbers:
            misclassified = False

    # check if there is a valid experience time period (base python)
    time_periods = {
        "week", "month", "year"
    }

    for time_period in time_periods:
        if bool(re.search(time_period, text.lower())):
            misclassified = False

    return misclassified

def update_misclassified_tags(input_data, output_data, iloc_span):
    """ updates the output data with correct tags """

    for i in range(iloc_span[0], iloc_span[1]+1):
        original_tag = str(input_data['tag'].iloc[i])

        # print(f"original tag:{original_tag}")

        if get_BIO_from_tag(original_tag) == 'B':
            new_tag = 'B-Skill'
            output_data['tag'].iloc[i] = new_tag
        elif get_BIO_from_tag(original_tag) == 'I':
            new_tag = 'I-Skill'
            output_data['tag'].iloc[i] = new_tag

        # print(f"new tag: {new_tag}\n")

    return output_data


def capture_misclassified_skills(input_data):
    """ uses regex to reassign misclassified Skills in Experience class """

    output_data = input_data.copy(deep=True)

    # initialise start and stop index to identify span
    iloc_span = [0,0]
    capture = False

    # iterate over rows in input data
    for row in input_data.itertuples():
        
        # if capture is off, and tag is B-Experience, set capture to True
        if not capture and row.tag == "B-Experience":
            capture = True
            iloc_span[0] = row.Index

        # if capture is on, and tag is not I-Experience:
        elif capture and row.tag != "I-Experience":
            capture = False
            iloc_span[1] = row.Index - 1

            # print(iloc_span)
            # print(input_data['word'].iloc[iloc_span[0]])
            # print(input_data['word'].iloc[iloc_span[1]])

            text = " ".join(list(input_data['word'].iloc[iloc_span[0]:iloc_span[1]+1]))

            # print(text)

            # identify if misclassified 
            if identify_misclassified_exp(text):
                # if misclassified, set tags in output_data with same index to B-Skill and I-Skill accordingly
                output_data = update_misclassified_tags(input_data, output_data, iloc_span)

    # if capture is on, check misclassification one more time (for final span)

    if capture:
        iloc_span[1] = len(input_data.index)
        # identify if misclassified 
        if identify_misclassified_exp(text):
            # if misclassified, set tags in output_data with same index to B-Skill and I-Skill accordingly
            output_data = update_misclassified_tags(input_data, output_data, iloc_span)

    return output_data

def split_spans_by_character(input_data, output_data, iloc_span, punctuation = {"/", "\\", ",", ".", ':', ';', '?', '!', '\/', '\,'}):
    """ splits spans by spcecial characters and reclassifies accordingly """

    try:
        span_dict = {
            x: input_data['word'].iloc[x] for x in range(iloc_span[0], iloc_span[1] + 1)
        }
    except:
        span_dict = {
            x: input_data['word'].iloc[x] for x in range(iloc_span[0], iloc_span[1])
        }

    special_character_indices = [
        index for index, value in span_dict.items() if value in punctuation
    ]

    # set tags of special characters to O
    # set BIO prefix of subsequent token (if one exists) to B
    for special_character_index in special_character_indices:
        output_data['tag'].iloc[special_character_index] = 'O'
        if special_character_index < iloc_span[1]:
            tag = get_class_from_tag(input_data['tag'].iloc[special_character_index + 1])
            if output_data['tag'].iloc[special_character_index + 1] != 'O':
                output_data['tag'].iloc[special_character_index + 1] = 'B-' + tag


    return output_data


def separate_terms(input_data):
    """ separates terms with special characters """

    output_data = input_data.copy(deep=True)

    # initialise start and stop index to identify span
    iloc_span = [0,0]
    current_tag = None
    capture = False

    # iterate over rows in input data
    for row in input_data.itertuples():

        prefix = get_BIO_from_tag(row.tag)
        tag = get_class_from_tag(row.tag)

        # if capture is off, and tag begins 'B', set capture to True and current_tag to current
        if not capture and prefix == 'B':

            capture = True
            current_tag = tag
            iloc_span[0] = row.Index

        # if capture is on, and tag is different to current_tag, close the span and capture
        elif capture and tag != current_tag:

            capture = False
            iloc_span[1] = row.Index - 1

            output_data = split_spans_by_character(input_data, output_data, iloc_span)

    # if capture is on, check current span one last time
    if capture:
        iloc_span[1] = len(input_data.index)
        output_data = split_spans_by_character(input_data, output_data, iloc_span)

    return output_data

def extra_preprocessing(input_data):
    """ combines above preprocessing into one function call """
    
    output_data = input_data.copy(deep=True)

    output_data = capture_misclassified_skills(output_data)
    output_data = separate_terms(output_data)

    return output_data