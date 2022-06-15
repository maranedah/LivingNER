team name: plncmm
affiliation: Center for Mathematical Modeling at University of Chile
mail: matias.rojas.g@ug.uchile.cl

authors:
	- Matías Rojas
	- Jose Barros
	- Mauricio Araneda
	- Jocelyn Dunstan

system explanation:


FLERT-Matcher 

- Subtask 1: We trained a NER model based on the FLERT approach. This model consists of fine-tuning a pre-trained language model
but considering the document-level context. 

- Subtask 2: Based on the predictions of subtask 1, we matched the entity mentions with the definitions of codes found in the training corpus and the NCBI taxonomy. This was done using the Levenshtein Distance.

- Subtask 3: We trained a NER model based on the FLERT approach to address each binary classification problem. We merged the output of each model with the predictions of subtask 2. Finally, we grouped the mentions by document and transformed the predictions to document-level.