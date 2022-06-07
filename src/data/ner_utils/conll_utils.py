import os 
import pandas as pd
from collections import defaultdict

def get_text_files_content(directory1, directory2):
    content = {}
    enc_directory1 = os.fsencode(directory1) 
    for file in os.listdir(enc_directory1):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            full_path = os.path.join(directory1, filename)
            name = filename.split('.')[0]
            content[name] = open(full_path, 'r', encoding = 'UTF-8').read()
    

    enc_directory2 = os.fsencode(directory2) 
    for file in os.listdir(enc_directory2):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            full_path = os.path.join(directory2, filename)
            name = filename.split('.')[0]
            content[name] = open(full_path, 'r', encoding = 'UTF-8').read()

    content = dict(sorted(content.items()))
    print(f'Number of documents found: {len(content)}')
    return content

def get_annotation_file_content(directory1, directory2):
    content = defaultdict(list)
    full_path = os.path.join(directory1, "training_entities_subtask1.tsv")
    df1 = pd.read_csv(full_path, sep='\t')
    for index, row in df1.iterrows():
        filename = row[0]
        start_index = int(row[3])
        end_index = int(row[4])
        label = row[2]
        span = row[5]
        content[filename].append({"start_index": start_index, "end_index": end_index, "label": label, "span": span})
    
    full_path = os.path.join(directory2, "validation_entities_subtask1.tsv")
    df2 = pd.read_csv(full_path, sep='\t')
    for index, row in df2.iterrows():
        filename = row[0]
        start_index = int(row[3])
        end_index = int(row[4])
        label = row[2]
        span = row[5]
        content[filename].append({"start_index": start_index, "end_index": end_index, "label": label, "span": span})

    content = dict(sorted(content.items()))
    return content

def get_annotation_file_content_subtask2(directory1, directory2):
    content = defaultdict(list)
    full_path = os.path.join(directory1, "training_entities_subtask2.tsv")
    df1 = pd.read_csv(full_path, sep='\t')
    for index, row in df1.iterrows():
       
        filename = row[0]
        start_index = int(row[3])
        end_index = int(row[4])
        label = row[2]
        span = row[5]

        if row[7] and label!='HUMAN': 
            content[filename].append({"start_index": start_index, "end_index": end_index, "label": "NOSOCOMIAL", "span": span})
    
    
    full_path = os.path.join(directory2, "validation_entities_subtask2.tsv")
    df2 = pd.read_csv(full_path, sep='\t')
    for index, row in df2.iterrows():
       
        filename = row[0]
        start_index = int(row[3])
        end_index = int(row[4])
        label = row[2]
        span = row[5]

        if row[7] and label!='HUMAN': 
            content[filename].append({"start_index": start_index, "end_index": end_index, "label": "NOSOCOMIAL", "span": span})
    
    
    content = dict(sorted(content.items()))
    return content

def get_annotation_file_content_subtask3(directory1, directory2, index, entity_type):
    content = defaultdict(list)
    full_path = os.path.join(directory1, "training_enriched_dataset_subtask3.tsv")
    df1 = pd.read_csv(full_path, sep='\t')
    for _, row in df1.iterrows():
       
        filename = row[0]
        start_index = int(row[3])
        end_index = int(row[4])
        label = row[2]
        span = row[5]

        if row[index]: 
            content[filename].append({"start_index": start_index, "end_index": end_index, "label": entity_type, "span": span})
    

    full_path = os.path.join(directory2, "validation_enriched_dataset_subtask3.tsv")
    df2 = pd.read_csv(full_path, sep='\t')
    for _, row in df2.iterrows():
       
        filename = row[0]
        start_index = int(row[3])
        end_index = int(row[4])
        label = row[2]
        span = row[5]

        if row[index]: 
            content[filename].append({"start_index": start_index, "end_index": end_index, "label": entity_type, "span": span})


    #n_entities = 0
    #for k, v in content.items():
    #    n_entities+=len(v)
    #print(f"Number of entity before reading subtask2 file: {n_entities}")


    #if index==9:
    #    directory = os.getcwd()
    #    directory = os.path.join(directory, 'training_valid_test_background/training')
    #    directory = os.path.join(directory, 'subtask2-Norm/')
    #    full_path = os.path.join(directory, "training_entities_subtask2.tsv")
    #    df = pd.read_csv(full_path, sep='\t')
    #    for _, row in df.iterrows():
    #    
    #        filename = row[0]
    #        start_index = int(row[3])
    #        end_index = int(row[4])
    #        label = row[2]
    #        span = row[5]

    #        if row[7]: 
    #            entity = {"start_index": start_index, "end_index": end_index, "label": entity_type, "span": span}
    #            if entity not in content[filename]:
                    #print(entity)
    #                content[filename].append(entity)
    
    #n_entities = 0
    #for k, v in content.items():
    #    n_entities+=len(v)
    #print(f"Number of entity after reading subtask2 file: {n_entities}")

    
    content = dict(sorted(content.items()))
    return content

def check_offsets(documents, annotations):
    for filename, content in documents.items():
        file_annotations = annotations[filename]
        for entity in file_annotations:
            start_index = entity["start_index"]
            end_index = entity["end_index"]
            span = entity["span"]
            original_span = content[start_index:end_index]
            if span != original_span:
                print("There is an inconsistency between the annotation indexes and the original text.")
                print("Filename: {}".format(filename))
                print("Original text: {} does not match with annotated entity {}".format(original_span, span))
                