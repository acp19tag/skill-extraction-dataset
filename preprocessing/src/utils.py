#! /usr/bin/python3
"""
handles the import of annotation data from Amazon Sagemaker / AMT and gold standard data in .csv
"""

import json
import pathlib
import hashlib
import ast
import glob
import os
import progressbar                              # pylint: disable=import-error
from tqdm import tqdm                           # pylint: disable=import-error

import pandas as pd                             # pylint: disable=import-error

import nltk                                     # pylint: disable=import-error
from nltk.tokenize import WordPunctTokenizer    # pylint: disable=import-error

class AnnotationSet():
    """
    class to handle import and preprocess of annotation data
    i.e. output from SageMaker/AMT

    ...

    """

    def __init__(self, data_dir: str):

        self.data_dir = data_dir
        self.annotation_list = self.csv_to_annotation_list(data_dir)

        self.reformatted_annotation_dict = self.reformat_annotation_dict(self.annotation_list)
        self.iaa_dict = self.generate_iaa_dict(self.reformatted_annotation_dict)
        self.dataframe = self.iaa_dict_to_dataframe(self.iaa_dict)
        self.gold_standard_worker_id = None

    def __str__(self):
        """ prints number of items in AnnotationSet and number of Workers """
        try:
            rows, columns = self.dataframe.shape
            return f"Data directory:\t{self.data_dir}\nTokens:\t\t{rows}\nWorkers:\t{columns-1}"
        except AttributeError:
            return "No dataframe exists."

    @staticmethod
    def check_filetype(data_dir: str) -> str:
        """ checks the file extension of an input file """
        return pathlib.Path(data_dir).suffix

    @staticmethod
    def csv_to_annotation_list(data_dir: str) -> list:
        """ converts a csv AMT output to annotation list """
        temp_df = combine_output_csv(data_dir)
        temp_df.columns = temp_df.columns.str.replace('.', '_') # fixes period issue
        first_pass_dict = {} # puts same id tasks in a dictionary together
        for row in temp_df.itertuples():
            if row.HITId not in first_pass_dict:
                first_pass_dict[row.HITId] = {
                    "content": row.Input_text,
                    "annotations": []
                }
            annotations = json.dumps(
                ast.literal_eval(row.Answer_taskAnswers)[0]
                ) # because amt puts this in a len1 list
            annotation_dict = {
                "workerId": row.WorkerId,
                "annotationData": {
                    "content": annotations
                }
            }
            first_pass_dict[row.HITId]["annotations"].append(annotation_dict)
        output_list = [] # put each task in its own dict, as per consolidated_items architecture
        for dataset_object_id in first_pass_dict:
            output_dict = {
                "datasetObjectId": dataset_object_id,
                "dataObject": {
                    "content": first_pass_dict[dataset_object_id]["content"]
                    },
                "annotations": first_pass_dict[dataset_object_id]["annotations"]
                }
            output_list.append(output_dict)
        return output_list

    @staticmethod
    def json_to_annotation_list(data_dir: str) -> list:
        """ reads annotations from json format to list of dictionaries """
        with open(data_dir) as infile:
            for segment in infile:
                output_string = segment
        return json.loads(output_string)

    @staticmethod
    def reformat_annotation_dict(annotation_list: list) -> dict:
        """ converts raw annotation list into one dictionary """

        def hash_content(content):
            """ returns sha1 encryption of content for id """
            return hashlib.sha1(str(content).encode('utf-8')).hexdigest()

        return {hash_content(item['dataObject']['content']): {
            'content':item['dataObject']['content'],
            'annotations':item['annotations']
            } for item in annotation_list}

    @staticmethod
    def generate_iaa_dict(reformatted_annotation_dict: dict, hit_list=None) -> dict:
        """ creates the inter-annotater agreement dictionary """

        def generate_span_dict(reformatted_annotation_dict: dict) -> dict:
            """ creates the token span dictionary """
            span_dict = {}
            for item_id in reformatted_annotation_dict:
                span_temp = reformatted_annotation_dict[item_id]['content']
                span_generator = WordPunctTokenizer().span_tokenize(span_temp)
                spans = [span for span in span_generator]
                span_dict[item_id] = spans
            return span_dict

        def generate_temp_annotation_dict(item_id: str, reformatted_annotation_dict: dict) -> dict:
            """ further reformat of annotation dict to line up with spans """
            temp_annotation_dict = {}
            for annotation_data in reformatted_annotation_dict[item_id]['annotations']:
                worker_id = annotation_data['workerId']
                temp_annotation_dict[worker_id] = {
                    (9999, 9999):'placeholder'
                    } # to fix a bug where 'none' labels are not entered
                for annotation in json.loads(
                        annotation_data['annotationData']['content']
                    )['crowd-entity-annotation']['entities']:
                    temp_annotation_dict[worker_id]\
                        [(annotation['startOffset'], annotation['endOffset'])]\
                        = annotation['label']
            return temp_annotation_dict

        def check_span_capture(annotation, span):
            """
            checks if a span is captured within an annotation
            (e.g. the token 'finance' in the annotation 'Finance Manager')
            """
            return annotation[0] <= span[0] and (annotation[1]+1) >= span[1]

        span_dict = generate_span_dict(reformatted_annotation_dict)
        token_index = 1
        iaa_dict = {}

        if hit_list:
            item_id_list = sorted([x for x in hit_list])
        else:
            item_id_list = sorted([x for x in reformatted_annotation_dict.keys()])

        for item_id in item_id_list:

            temp_annotation_dict = generate_temp_annotation_dict(
                item_id, reformatted_annotation_dict
                )

            for span in span_dict[item_id]:
                if token_index not in iaa_dict.keys(): # add to iaa_dict if not already there
                    iaa_dict[token_index] = {
                        'text': reformatted_annotation_dict[item_id]['content'][span[0]:span[1]]
                        }
                for worker_id in temp_annotation_dict:
                    match = False
                    for annotation in temp_annotation_dict[worker_id]:
                        if check_span_capture(annotation, span):
                            iaa_dict[token_index][worker_id] \
                                = temp_annotation_dict[worker_id][annotation]
                            match = True
                        if not match: # if no match is found, insert a 'NONE' label
                            iaa_dict[token_index][worker_id] = 'NONE'
                token_index += 1
        return iaa_dict

    @staticmethod
    def iaa_dict_to_dataframe(iaa_dict: dict, gold_standard_worker_id=None) -> pd.DataFrame:
        """ converts inter-annotator agreement dictionary into a pandas dataframe """
        if gold_standard_worker_id:
            df_temp = pd.DataFrame(iaa_dict).T
            return df_temp[['text', gold_standard_worker_id]]
        return pd.DataFrame(iaa_dict).T

def import_gold_standard(data_dir: str, gold_standard_worker_id: str) -> AnnotationSet:
    """ returns an annotationset with the gold_standard worker id """
    output = AnnotationSet(data_dir)
    output.gold_standard_worker_id = gold_standard_worker_id
    return output

def import_labels(data_dir: str, o_label=None) -> list:
    """ returns a list of all the positive labels with a 'none' label if specified """
    with open(data_dir) as infile:
        labels_list = [line.rstrip() for line in infile]
    if o_label:
        labels_list.append(o_label)
    return labels_list

def combine_output_csv(directory):
        """ concatenates all .csv files in a directory """
        all_files = glob.glob(os.path.join(directory, "*.csv"))
        return pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

def get_worker_list(AnnotationSet):
    """ return a list of worker IDs """
    return list(AnnotationSet.dataframe.columns)[1:]

def add_label_prefix(worker, response_dict, previous_label_dict):
    """ adds the 'B' or 'I' prefix to the label """
    label = response_dict[worker]
    previous_label = previous_label_dict[worker]
    ignore_set = {'NONE', 'O', '?'}

    if label in ignore_set:
        return ''
    if label == previous_label:
        return 'I-'
    return 'B-'

def update_previous_label(response_dict, worker_list):
    """ returns a new dictionary showing the previous label for each worker """
    previous_label_dict = {worker:'?' for worker in worker_list}
    if response_dict['text'] != '"':
        for worker in response_dict:
            if worker in previous_label_dict:
                previous_label_dict[worker] = response_dict[worker]
    return previous_label_dict

def dataset_row_constructor(response_dict, worker_list, previous_label_dict, BO_tags, labels_to_suppress):
    """ creates a single dataset row from one ID from iaa_dict """
    original_text = response_dict['text']
    if original_text == '"':
        return '\n'
    reformatted_text = original_text.replace('"', '')
    line_str = reformatted_text + " "
    for worker in worker_list:
        if worker in response_dict:
            if response_dict[worker] == 'NONE' or response_dict[worker] in labels_to_suppress:
                line_str += "O "
            elif BO_tags:
                line_str += add_label_prefix(worker, response_dict, previous_label_dict) + response_dict[worker] + " "
            else:
                line_str += response_dict[worker] + " "
        else:
            line_str += "? "
    if original_text[0] == '"':
        return '\n' + line_str + '\n'
    elif original_text[-1] == '"':
        return line_str + '\n' + '\n'
    return line_str + '\n'

def AnnotationSet_to_dataset_format(AnnotationSet, output_dir, BO_tags = False, labels_to_suppress = []):
    """ converts an annotationset to the correct format for the finished dataset """
    iaa_dict = AnnotationSet.iaa_dict
    worker_list = get_worker_list(AnnotationSet)
    previous_label_dict = {worker:'?' for worker in worker_list}

    with open(output_dir, 'w') as f:
        pass
    
    for i in progressbar.progressbar(range(1, len(iaa_dict))): 
        with open(output_dir, 'a') as f:
            f.write(dataset_row_constructor(iaa_dict[i], worker_list, previous_label_dict, BO_tags, labels_to_suppress))
        previous_label_dict = update_previous_label(iaa_dict[i], worker_list)

def save_worker_list(AnnotationSet, output_dir):
    """ saves the list of workers for reference/weighting """
    with open(output_dir, 'w') as f:
        for worker in get_worker_list(AnnotationSet):
            f.write(worker)
            f.write('\n')

def show_worker_distribution(AnnotationSet):
    """ shows the number of annotated tokens by Worker ID """
    worker_list = get_worker_list(AnnotationSet)
    return {
        workerID: AnnotationSet.dataframe[workerID].count() for workerID in worker_list
    }

def find_tag(tag_list, method, priority_labels=[], worker_indices=[]):
    """ given a list of tags, return the appropriate one for the selection method """

    if method == 'first':
        # remove ? tags
        tag_list = [x for x in tag_list if x != '?']
        return tag_list[0]

    elif method == 'last':
        # remove ? tags
        tag_list = [x for x in tag_list if x != '?']
        return tag_list[-1]

    elif method == 'priority_labels':
        for tag in priority_labels:
            if tag in tag_list:
                return tag
            return None

    elif method == 'worker_indices':
        for worker_index in worker_indices:
            tag = tag_list[worker_index]
            if tag != '?':
                return tag

def get_ranked_worker_indices(worker_accuracies_dir):
    """ returns a list of indices in decreasing priority,
    taken from ranking worker accuracies from qualification task """

    worker_accuracies_df = pd.read_csv(worker_accuracies_dir)

    return list(worker_accuracies_df.sort_values(by=['accuracy'], ascending=False)['worker_index'])


def txt_to_df(file_dir, method, priority_list=[], worker_indices=[]):
    """ creates a dataframe from txt training data """

    sentence_number = 1
    sentence_number_list = []
    word_list = []
    pos_list = []
    tag_list = []
    rolling_sentence_list = []
    previous_line_break = True

    num_lines = sum(1 for line in open(file_dir))
    with open(file_dir) as f:
        for line in tqdm(f, total=num_lines):
            if line.split(): # i.e. not line break
                line_items = line.split()
                word = line_items[0]
                current_tag_list = line_items[1:]

                sentence_number_list.append(str(sentence_number))
                word_list.append(word)
                tag_list.append(find_tag(current_tag_list, method, priority_list, worker_indices))
                rolling_sentence_list.append(word)

                previous_line_break = False

            elif not previous_line_break:
                previous_line_break = True
                # DO SENTENCE POS TAGGING
                for tag in [x[1] for x in nltk.pos_tag(rolling_sentence_list)]:
                    pos_list.append(tag)
                rolling_sentence_list = []
                sentence_number += 1
    # one more time for the last one...
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

# EXTRA PREPROCESSING - utils

def suppress_classes(label, list_to_suppress):
    """ returns 'O' if class in list """
    if label in list_to_suppress:
        return 'O'
    return label

def update_entity_dict(entity_dict, span_capture_list, previous_entity):
    """ update the entity dict by unloading the capture list """

    new_entity_dict = entity_dict.copy()

    if previous_entity and span_capture_list:

        span = TreebankWordDetokenizer().detokenize(span_capture_list)

        if previous_entity not in new_entity_dict:

            new_entity_dict[previous_entity] = [span]
    
        else:

            new_entity_dict[previous_entity].append(span)

    # reset the rolling variables
    span_capture_list = []
    previous_entity = None 

    return new_entity_dict, span_capture_list, previous_entity



def parse_row(row, entity_dict, span_capture_list, previous_entity):
    """ updates the entity dict and span capture list based on row contents """

    bio_tag, entity = parse_tag(row.tag)

    if bio_tag == 'B':
        
        # update with previous entity, if applicable
        entity_dict, span_capture_list, previous_entity = update_entity_dict(entity_dict, span_capture_list, previous_entity)
        
        # start collecting new entity
        span_capture_list = [row.word]
        previous_entity = entity

    elif bio_tag == 'I':
        
        # continue collecting entity
        span_capture_list.append(row.word)

    else:
        
        # update with previous entity, if applicable
        entity_dict, span_capture_list, previous_entity = update_entity_dict(entity_dict, span_capture_list, previous_entity)
        previous_entity = None

    return entity_dict, span_capture_list, previous_entity


def get_entity_dict(dataframe):
    """ returns a dictionary of all entity mentions in dataframe """
    
    entity_dict = {}
    span_capture_list = []
    previous_entity = None

    for row in dataframe.itertuples():

        entity_dict, span_capture_list, previous_entity = parse_row(row, entity_dict, span_capture_list, previous_entity)

    # capture the last entity, if applicable
    entity_dict, span_capture_list, previous_entity = update_entity_dict(entity_dict, span_capture_list, previous_entity)

    return entity_dict

def print_entity_dict_to_file(entity_dict, output_dir):

    """ prints each key in entity dict to separate file """

    for key in entity_dict:

        filename = output_dir + key + ".txt"

        with open(filename, 'w') as f:

            f.write("\n".join(entity_dict[key]))

def entity_dict_to_list(entity_dict, entity_type='all'):
    """ returns a list of entities from the entity dict """

    if entity_type == 'all':

        output_list = []

        for classification in entity_dict:

            for item in entity_dict[classification]:

                output_list.append(item)

        return output_list