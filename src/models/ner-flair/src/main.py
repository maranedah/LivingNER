import yaml 
import torch
import flair 
from datasets import NERCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings

if __name__=='__main__':
    
    # Read configuration file
    with open('../config.yaml') as file:
        config = yaml.safe_load(file)
    
    print('======================== Parameters ========================')
    for k, v in config.items():
        print(f'{k}: {v}')
    print('============================================================')

    available_gpu = torch.cuda.is_available()
    
    if available_gpu:
        flair.device = torch.device('cuda')
    else:
        flair.device = torch.device('cpu')
    
    flair.set_seed(config["seed"])

    corpus = NERCorpus(config["data_folder"]).create_corpus()
    print(corpus)
    # print the first Sentence in the training split
    label_type = 'ner'

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    print(label_dict)

    if config["model_type"]=="roberta":
    # 4. initialize fine-tuneable transformer embeddings WITH document context
        embeddings = TransformerWordEmbeddings(model=config["model_name"],
                                            layers="-1" if config["fine_tune"] else "all",
                                            subtoken_pooling=config["subtoken_pooling"],
                                            fine_tune=config["fine_tune"],
                                            use_context=True,
                                            )
    
    if config["model_type"]=="flair":
        embedding_types = [
            FlairEmbeddings('es-clinical-forward'),
            FlairEmbeddings('es-clinical-backward'),
        ]

        embeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
    tagger = SequenceTagger(hidden_size=config["hidden_size"],
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type='ner',
                            use_crf=False if config["fine_tune"] else True,
                            use_rnn=False if config["fine_tune"] else True,
                            reproject_embeddings=False if config["fine_tune"] else True,
                            rnn_layers= config["rnn_layers"]
                            )

    if not config["start_from_checkpoint"]: 

        # 6. initialize trainer
        trainer = ModelTrainer(tagger, corpus)

        if config["fine_tune"]:

            # 7. run fine-tuning
            trainer.fine_tune(config["output_path"],
                            learning_rate=5e-06,
                            mini_batch_size=config["mini_batch_size"],
                            mini_batch_chunk_size=config["mini_batch_chunk_size"],  # remove this parameter to speed up computation if you have a big GPU
                            train_with_dev=config["train_with_dev"],
                            train_with_test=config["train_with_test"],
                            checkpoint=True,
                            max_epochs = config["max_epochs"]
                            )
        
        else: 
             trainer.train(config["output_path"],
                            learning_rate=config["learning_rate"],
                            mini_batch_size=config["mini_batch_size"],
                            train_with_dev=config["train_with_dev"],
                            train_with_test=config["train_with_test"],
                            checkpoint=True,
                            max_epochs = config["max_epochs"]
                            )


    else:

        trainer: ModelTrainer = ModelTrainer(
                model = tagger, 
                corpus = corpus)

        trained_model = SequenceTagger.load(f'{config["output_path"]}/checkpoint.pt')

        # resume training best model, but this time until epoch 25
        trainer.resume(trained_model,
                    base_path=config["output_path"],
                    max_epochs=config["max_epochs"],
                    )