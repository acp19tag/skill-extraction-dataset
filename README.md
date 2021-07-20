# Skill Extraction Dataset

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
| data | answers.txt | Worker responses from the annotation task on AMT. |
| data | testset.txt | Gold standard dataset for model evaluation. |
| data | answers_worker_list.txt | List of Worker IDs according to their response index in answers.txt. |
| data | worker_accuracies.csv | Dataframe of the accuracies of each Worker on the evaluation task (>70% required for participation). |

## License

Creative Commons Zero v1.0 Universal