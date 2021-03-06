# Benchmark Corpus to Support Entity Recognition in Job Descriptions

## Overview

A public, human-labelled dataset of salient entities in job descriptions. 

Salient entities include: 
- Skill
- Qualification
- Experience
- Occupation
- Domain

Full details regarding annotation schema can be found in `schema/Combined_Annotation_Instructions.pdf`.

Labels were collected from [Amazon Mechanical Turk](https://www.mturk.com/). Workers were required to achieve >70% accuracy in a qualification task before contributing to this dataset. 

Data formatting follows [2003 CONLL NER dataset](https://www.clips.uantwerpen.be/conll2003/ner/) conventions. Individual Worker responses, Worker ID and associated accuracies on the qualification task have been retained. 

Original job description data can be found at [Kaggle](https://www.kaggle.com/airiddha/trainrev1). Credit to user [airiddha](https://www.kaggle.com/airiddha) for the development of the original dataset. 

## File Descriptions

| Folder | Filename | Description
| ------ | ------ | ------ |
| raw_data | answers.txt | Worker responses from the annotation task on AMT. |
| raw_data | testset.txt | Gold standard dataset for model evaluation. |
| raw_data | worker_accuracies.csv | Dataframe of the accuracies of each Worker (identified by their index in answers.txt) on the evaluation task (>70% required for participation). |
| preprocessing | data_aggregation.py | Python script for aggregating worker answers into single label pandas dataframe. |
| preprocessing | extra_preprocessing.py | Python script for simple data preprocessing. Used in baseline model. |
| baseline | train_baseline_crf.py | Python script that trains, saves, and evaluates the baseline CRF model for entity recognition. |
| baseline | load_baseline_crf.py | Python script that loads the saved baseline model and offers an interactive prompt for classification. train_baseline_crf must be run first. |
| schema | Combined_Annotation_Instructions.pdf | Schema document used during data collection on Amazon Mechanical Turk. |

## License

Creative Commons Attribution 4.0 International
