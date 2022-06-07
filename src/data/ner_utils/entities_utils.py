from collections import defaultdict

def colapse_with_others(entity_info, entities):
    for entity in entities:
        if (entity_info["start_index"]<entity["start_index"] and entity_info["end_index"]<entity["end_index"] and entity_info["end_index"]>entity["start_index"])\
            or (entity_info["start_index"] > entity["start_index"] and entity_info["start_index"] < entity["end_index"] and entity_info["end_index"] > entity["end_index"]):
            return True
    return False

def filter_crossing_entities(annotations):
  filtered_annotations = defaultdict(list)
  for filename, entities in annotations.items():
    added_entities = []
    for entity in entities:
      if not colapse_with_others(entity, added_entities):
        filtered_annotations[filename].append(entity)
        added_entities.append(entity)
  return filtered_annotations

def simplify_nested_entities(annotations):
    flat_annotations = defaultdict(list)
    for filename, entities in annotations.items():
        nestings = get_nestings(entities)   
  
        for entity in entities:
          is_nested = False
          for n in nestings:
              if entity in n[1:]:
                  is_nested=True
          if not is_nested:
   
            flat_annotations[filename].append(entity)
          

    return flat_annotations

def get_nestings(entities):
  nestings = [] 

  for entity_1 in entities:

    is_outer = True 
    possible_nested_entity = [entity_1]

    for entity_2 in entities:
      if entity_1!=entity_2:

        entity_1_start_index = entity_1["start_index"]
        entity_1_end_index = entity_1["end_index"]

        entity_2_start_index = entity_2["start_index"]
        entity_2_end_index = entity_2["end_index"]

        if (entity_1_start_index >= entity_2_start_index and entity_1_end_index <= entity_2_end_index):
            is_outer = False 

        if (entity_2_start_index >= entity_1_start_index and entity_2_end_index <= entity_1_end_index):
            possible_nested_entity.append(entity_2)
 
    
    if is_outer and len(possible_nested_entity)>1:
      possible_nested_entity.sort(key=lambda entity: (entity["end_index"]-entity["start_index"], entity["label"]), reverse=True)
      if possible_nested_entity not in nestings:
        nestings.append(possible_nested_entity)
  return nestings