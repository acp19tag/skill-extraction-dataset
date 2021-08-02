#! /usr/bin/python3
"""
handles the construction of the dataframe from dataset/testset files
"""

from src.utils import * # pylint: disable=all

answers_file_dir = '../raw_data/answers.txt'
answers_outfile_dir = '../aggregated_data/df_answers.csv'

testset_file_dir = '../raw_data/testset.txt'
testset_outfile_dir = '../aggregated_data/df_testset.csv'

ranked_worker_indices = get_ranked_worker_indices(
    worker_accuracies_dir = '../raw_data/worker_accuracies.csv'
    )

df = txt_to_df(
    answers_file_dir, 
    method='worker_indices', 
    priority_list=[], 
    worker_indices = ranked_worker_indices
    )
df.to_csv(answers_outfile_dir, index=False)

df = txt_to_df(
    testset_file_dir, 
    method='first', 
    priority_list=[]
    )
df.to_csv(testset_outfile_dir, index=False)