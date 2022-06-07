import os 
import spacy
import re
import math
import random
from conll_utils import get_text_files_content, get_annotation_file_content, get_annotation_file_content_subtask2, get_annotation_file_content_subtask3, check_offsets
from entities_utils import simplify_nested_entities, filter_crossing_entities
from tqdm.auto import tqdm
from collections import defaultdict
random.seed(123)

nlp = spacy.load("es_core_news_lg")


def create_partitions(output_directory, entity_type):

    f = open(f'{output_directory}/{entity_type}.conll', 'r').read()
    f = re.sub(r'\n\s*\n', '\n\n', f)
    annotations = f.split('\n\n')
    random.shuffle(annotations)
    n_examples = len(annotations)
    n_train = math.floor(n_examples*0.6)
    n_val =  math.floor(n_examples*0.2)
    n_test=  math.floor(n_examples*0.2)
    train = open(f'{output_directory}/{entity_type}_train.iob2', 'w')
    for i in range(0, n_train):
        if i!=n_train-1: train.write(annotations[i] + "\n\n")
        else: train.write(annotations[i])
    train.close() 

    dev = open(f'{output_directory}/{entity_type}_valid.iob2', 'w')
    for i in range(n_train, n_train+n_val):
        if i!=n_train+n_val-1: dev.write(annotations[i] +"\n\n")
        else: dev.write(annotations[i])
    dev.close()

    test = open(f'{output_directory}/{entity_type}_test.iob2', 'w')
    for i in range(n_train+n_val, n_examples):
        if i!=n_examples-1: test.write(annotations[i] +"\n\n")  
        else: test.write(annotations[i])  
    test.close() 
  

def create_conll_file(documents, annotations, output_directory, entity_type):
    output_file = open(f'{output_directory}/{entity_type}.conll', 'w')
    original_entities_count = sum(len(v) for k, v in annotations.items())
    annotations = filter_crossing_entities(annotations)
    flat_annotations = simplify_nested_entities(annotations)
    flat_entities_count = sum(len(v) for k, v in flat_annotations.items())
    print(f'Original number of entities in partition: {original_entities_count}. {original_entities_count - flat_entities_count} deleted by nestings and crossing entities.')

    entities_annotated = 0
    for filename, content in tqdm(documents.items()):
        doc = nlp(content)
        entities = flat_annotations[filename]
        entities_added = []
        for sent in doc.sents:

            for token in sent:

                assert(token.text==content[token.idx:token.idx+len(token)])
                if not token.text or '\n' in token.text or '\t' in token.text or token.text.strip()=='':
                    continue
                
                token_tag = 'O'
                token_start = token.idx
                token_end = token.idx + len(token)

                for entity in entities:
                    if token_start == entity["start_index"]:
                        token_tag = f'B-{entity["label"]}'
                        entities_annotated+=1
                        entities_added.append(entity)
                        break
                    elif token_start > entity["start_index"] and token_end <= entity["end_index"]:
                        token_tag=f'I-{entity["label"]}'
                        break
                    else:
                        pass
                
                output_file.write(f'{token.text}\t{token_tag}\n')
            output_file.write('\n')
            

    print(entities_annotated)
    pass

def create_data_for_subtask_1(output_directory):


    directory = os.getcwd()

    training_directory = os.path.join(directory, 'training_valid_test_background/training')
    training_text_files_directory = os.path.join(training_directory, 'text-files/')
    training_ann_directory = os.path.join(training_directory, 'subtask1-NER/')

    valid_directory = os.path.join(directory, 'training_valid_test_background/valid')
    valid_text_files_directory = os.path.join(valid_directory, 'text-files/')
    valid_ann_directory = os.path.join(valid_directory, 'subtask1-NER/')


    # Processing text and ann files
    text_files_content = get_text_files_content(training_text_files_directory, valid_text_files_directory)
    ann_content = get_annotation_file_content(training_ann_directory, valid_ann_directory)

    # Splitting data into train, valid, and test (0.6, 0.2, 0.2)


    documents = text_files_content
    annotations = ann_content
    check_offsets(documents, annotations)
    create_conll_file(documents, annotations, output_directory, 'subtask1')
    create_partitions(output_directory, 'subtask1')

   

def create_data_for_subtask_2(output_directory):
    directory = os.getcwd()
    training_directory = os.path.join(directory, 'training_valid_test_background/training')
    training_text_files_directory = os.path.join(training_directory, 'text-files/')
    training_ann_directory = os.path.join(training_directory, 'subtask2-Norm/')

    valid_directory = os.path.join(directory, 'training_valid_test_background/valid')
    valid_text_files_directory = os.path.join(valid_directory, 'text-files/')
    valid_ann_directory = os.path.join(valid_directory, 'subtask2-Norm/')

    # Processing text and ann files
    text_files_content = get_text_files_content(training_text_files_directory, valid_text_files_directory)
    ann_content = get_annotation_file_content_subtask2(training_ann_directory, valid_ann_directory)

    text_files_content_filtered = defaultdict(list)
    for filename, content in text_files_content.items():
      if filename in ann_content:
        text_files_content_filtered[filename]=content
    text_files_content = dict(sorted(text_files_content_filtered.items()))
    
    documents = text_files_content
    annotations = ann_content
    check_offsets(documents, annotations)
    create_conll_file(documents, annotations, output_directory, 'NOSOCOMIAL')
    create_partitions(output_directory, 'NOSOCOMIAL')


def create_data_for_subtask_3(output_directory, entity_type='pet'):
    # This function creates the formatted data to train the NER models for subtask 3.
    # It receives the output directory to store the files, and also the parameters entity_type
    # which specifies the category.
    directory = os.getcwd()

    training_directory = os.path.join(directory, 'training_valid_test_background/training')
    training_text_files_directory = os.path.join(training_directory, 'text-files/')
    training_ann_directory = os.path.join(training_directory, 'subtask3-Clinical-Impact/')

    valid_directory = os.path.join(directory, 'training_valid_test_background/valid')
    valid_text_files_directory = os.path.join(valid_directory, 'text-files/')
    valid_ann_directory = os.path.join(valid_directory, 'subtask3-Clinical-Impact/')

    text_files_content = get_text_files_content(training_text_files_directory, valid_text_files_directory)

    if entity_type == "PET":
        index = 6
        pass
    
    if entity_type == "ANIMALINJURY":
        index = 7
        pass
    
    if entity_type == "FOOD":
        index = 8
        pass
    
    if entity_type == "NOSOCOMIAL":
        index = 9
        pass

    ann_content = get_annotation_file_content_subtask3(training_ann_directory, valid_ann_directory, index, entity_type)
    text_files_content_filtered = defaultdict(list)
    for filename, content in text_files_content.items():
      if filename in ann_content:
        text_files_content_filtered[filename]=content
    text_files_content = dict(sorted(text_files_content_filtered.items()))

    documents = text_files_content
    annotations = ann_content
    check_offsets(documents, annotations)
    create_conll_file(documents, annotations, output_directory, entity_type)
    create_partitions(output_directory, entity_type)


    pass