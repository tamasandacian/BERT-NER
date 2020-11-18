import transformers
from transformers import BertForTokenClassification, BertTokenizer
from normalization import TextNormalization
from shared.utils import load_from_json
from shared.utils import isFile
from functools import lru_cache
from utility import Utility
import numpy as np
import logging
import torch
import torch.nn.functional as F
import os

device = torch.device("cpu")
abs_path = os.path.abspath(os.path.dirname(__file__)) + "/output"
label_desc_path = os.path.abspath(os.path.dirname(__file__)) + "/label_desc"

class Entity(object):
    """ Class to store Entity metadata
    
    :param token: entity token/word
    :param label: entity label
    :param definition: entity label definition
    :param confidence: entity probability in text
    :param index: token index position in text
    """
    def __init__(self, token, label, definition, confidence, index):
        self.token = token
        self.label = label
        self.definition = definition
        self.confidence = confidence
        self.index = index

class NER(object):
    """ Class for predicting entities in new documents
    
    :param lang_code: language model
    :param version: model version number
    :param clean_text: boolean flag for cleaning text
    :param min_words: min number of words for prediction
    :param min_conf_score: minimum confidence threshold
    :param max_length: max sequence length required by BERT pre-trained model
    :param pre_trained_name: pre-trained model name
    """
    def __init__(self, lang_code, version="1.1", min_words=10, min_conf_score=0.20, max_length=128, clean_text=False, 
                 pre_trained_name='bert-base-cased'):
        
        self.lang_code = lang_code
        self.version = version
        self.min_words = min_words
        self.min_conf_score = min_conf_score
        self.max_length = max_length
        self.clean_text = clean_text
        self.pre_trained_name = pre_trained_name
        self.label_desc = dict()
        
        self.model_path = abs_path + "/models"
        self.label_index_path = abs_path + "/label_index"
        subdir = "{}_{}".format(self.lang_code, self.version)

        self.valid_langs = ["en"]
        if lang_code in self.valid_langs:
            self.model_path = self.model_path + "/" + subdir + "/model.pt"
            self.label_index_path = self.label_index_path + "/" + subdir + "/label_index.json"
            self.label_desc_path = label_desc_path + "/" + lang_code + "/label_desc.json"
            
            if isFile(self.model_path) and isFile(self.label_index_path):
                self.label_desc = load_from_json(self.label_desc_path)
                self.label_index = load_from_json(self.label_index_path)
                self.index_label = {v: k for k, v in self.label_index.items()}
                self.tokenizer = self.load_tokenizer(pre_trained_name)
                self.tn = TextNormalization(self.tokenizer)
                self.model = self.load_model(pre_trained_name, num_labels=len(self.label_index))
                self.model.to(device)
                self.model.load_state_dict(torch.load(self.model_path, device), strict=False)

    @lru_cache(maxsize=128)
    def load_model(self, pre_trained_name, num_labels):
        """ Load BERT pre-trained model with given num labels

        :param pre_trained_name: BERT pre-trained model name
        :param num_labels: total number of labels
        :return: BertForTokenClassification model
        """
        model = BertForTokenClassification.from_pretrained(
            pre_trained_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        return model

    @lru_cache(maxsize=128)
    def load_tokenizer(self, pre_trained_name):
        """ Load BERT pre-trained tokenizer

        :param pre_trained_name: BERT pre-trained model name
        :return: BERT tokenizer
        """
        tokenizer = BertTokenizer.from_pretrained(pre_trained_name)
        return tokenizer

    def predict_entities(self, text):
        """ Predict entities in text

        :param text
        :return: dictionary
        """
        try:
            prediction = dict()

            if text:
                if Utility.get_doc_length(text) > self.min_words:
                    if self.lang_code in self.valid_langs:
                        if self.lang_code == "en":
                            if self.clean_text:
                                text = Utility.clean_text(text)

                            if isFile(self.model_path) and isFile(self.label_index_path):
                                
                                # pre-process text using BERT tokenizer
                                text_input_ids, attention_masks = self.tn.text_preprocessing(text)
                                
                                # get model weights
                                with torch.no_grad():
                                    outputs = self.model(text_input_ids, attention_masks)

                                # get probabilities using Softmax function
                                logits = F.softmax(outputs[0], dim=2)    
                                # get label indices
                                label_ids = torch.argmax(logits, dim=2)
                                label_ids = label_ids.numpy()[0]
                                
                                # get score indices
                                conf_indices = [values[label].item() for values, label in zip(logits[0], label_ids)]
                                labels = [self.index_label[i] for i in label_ids]
                                text_tokens = self.tokenizer.convert_ids_to_tokens(text_input_ids.numpy()[0])

                                entities = []
                                for index, (token, label, confidence) in enumerate(zip(text_tokens, labels, conf_indices)):    
                                    definition = self.label_desc[label]
                                    if label == "B-per" or label == 'I-per':
                                        entity = Entity(token, label, definition, confidence, index)
                                        entities.append(entity.__dict__)                        
                                    elif label == 'B-tim' or label == 'I-tim':
                                        entity = Entity(token, label, definition, confidence, index)
                                        entities.append(entity.__dict__)
                                    elif label == 'B-org' or label == 'I-org':
                                        entity = Entity(token, label, definition, confidence, index)
                                        entities.append(entity.__dict__)
                                    elif label == 'B-nat' or label == 'I-nat':
                                        entity = Entity(token, label, definition, confidence, index)
                                        entities.append(entity.__dict__)
                                    elif label == 'B-art' or label == 'I-art':
                                        entity = Entity(token, label, definition, confidence, index)
                                        entities.append(entity.__dict__)
                                    elif label == 'B-geo' or label == 'I-geo':
                                        entity = Entity(token, label, definition, confidence, index)
                                        entities.append(entity.__dict__)
                                    elif label == 'B-gpe' or label == 'I-gpe':
                                        entity = Entity(token, label, definition, confidence, index)
                                        entities.append(entity.__dict__)
                                    elif label == 'B-eve' or label == 'I-eve':
                                        entity = Entity(token, label, definition, confidence, index)
                                        entities.append(entity.__dict__)

                                prediction["entities"]  = entities
                                prediction["message"] = 'successful'
                                return prediction
                            else:
                                return "model not found"
                    else:
                        return "language not supported"
                else:
                    return "required at least {} words for extraction".format(self.min_words)            
            else:
                return "required textual content"              
        except Exception:
            logging.error('exception occured', exc_info=True)



