#! /usr/bin/python3
"""
generates insights about annotated data such as Worker agreement, accuracy given gold standard
"""
import itertools                # pylint: disable=import-error

import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
import sklearn                  # pylint: disable=import-error
import matplotlib.pyplot as plt # pylint: disable=import-error

def find_matching_hits(annotation_set_1, annotation_set_2):
    """ returns a list of keys for hits in both annotationsets """
    return list(
        set(
            annotation_set_1.reformatted_annotation_dict.keys()) \
            .intersection(
                set(
                    annotation_set_2.reformatted_annotation_dict.keys()
                    )
                )
        )

def filter_for_matching_hits(annotation_set, gold_standard):
    """ returns two dataframes containing only hits present in the other """
    intersection_list = find_matching_hits(annotation_set, gold_standard)
    annotation_set_iaa_dict = annotation_set.generate_iaa_dict(
        annotation_set.reformatted_annotation_dict,
        intersection_list
        )
    annotation_set_dataframe = annotation_set.iaa_dict_to_dataframe(
        annotation_set_iaa_dict
        )
    gold_standard_iaa_dict = gold_standard.generate_iaa_dict(
        gold_standard.reformatted_annotation_dict,
        intersection_list
        )
    gold_standard_dataframe = gold_standard.iaa_dict_to_dataframe(
        gold_standard_iaa_dict,
        gold_standard.gold_standard_worker_id
        )
    return annotation_set_dataframe, gold_standard_dataframe

def get_worker_list(annotation_dataframe):
    """ return a list of worker IDs """
    return list(annotation_dataframe.columns)[1:]

def get_worker_set(annotation_dataframe, workers_to_suppress=None):
    """ return a list of worker IDs, suppressing those specified """
    if workers_to_suppress:
        return set(get_worker_list(annotation_dataframe)) - set(workers_to_suppress)
    else:
        return set(get_worker_list(annotation_dataframe))

def combine_dataframes_without_na(dataframe_1, dataframe_2):
    """ combines two dataframes side-by-side and drops na rows """
    return pd.concat([dataframe_1, dataframe_2], axis=1).dropna()

def generate_agreement_grid(annotation_dataframe, labels_list):
    """ returns a Dataframe showing kappa agreement between workers """
    worker_list = get_worker_list(annotation_dataframe)
    data_dict = {}
    for worker_a in worker_list:
        for worker_b in worker_list:
            if worker_a == worker_b: # if annotators are the same - perfect agreement
                agreement = 1
            else:
                selected_columns = annotation_dataframe[[worker_a, worker_b]]
                df_copy = selected_columns.copy()
                df_copy = df_copy.dropna()
                if df_copy.empty:
                    # agreement = 'NA'
                    agreement = NaN
                else:
                    agreement = sklearn.metrics.cohen_kappa_score(
                        df_copy[worker_a],
                        df_copy[worker_b],
                        labels=labels_list)
            if worker_a not in data_dict:
                data_dict[worker_a] = [agreement]
            else:
                data_dict[worker_a].append(agreement)

    return pd.DataFrame(data_dict, columns=worker_list, index=worker_list)

def calculate_agreement(annotation_dataframe, labels_list, workers_to_suppress=None):
    """ calculates the agreement between workers in the annotation data """
    # worker_set = get_worker_set(annotation_dataframe, workers_to_suppress)

    # agreement_list = [
    #     sklearn.metrics.cohen_kappa_score(
    #         annotation_dataframe[worker_a],
    #         annotation_dataframe[worker_b],
    #         labels=labels_list,
    #     )
    #     for worker_a, worker_b in itertools.combinations(worker_set, 2)
    # ]
    # return np.mean(agreement_list)

    agreement_grid = generate_agreement_grid(annotation_dataframe, labels_list)
    worker_set = get_worker_set(annotation_dataframe, workers_to_suppress)
    agreement_list = [
        agreement_grid.loc[worker_a, worker_b] for worker_a, worker_b in itertools.combinations(worker_set, 2)
    ]
    return np.nanmean(agreement_list)


