import torch

class TextNormalization(object):
    """ Class for pre-processing textual content using BERT tokenizer
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_text(self, text):
        """ Tokenize text using BERT tokenizer
        
        :param text: text
        :return: tokenized text
        """
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokenize_pair(self, text_tokens, text_labels):
        """ Tokenize tokens in text and preserve labels using BERT tokenizer

        :param text_tokens: text tokens
        :param text_labels: text labels
        :return: tuple (tokens, labels) in text
        """

        tokens = []
        labels = []
        for token, label in zip(text_tokens, text_labels):
              # tokenize word
              token = self.tokenizer.tokenize(token)
              # get total number of subwords from tokenized word
              num_subwords = len(token)
              # get equal sized list of tokens, labels in a sentence
              tokens.extend(token)
              labels.extend([label] * num_subwords)

        return tokens, labels

    def convert_tokens_to_ids(self, tokens):
        """ Convert list of tokens to input ids
        
        :param tokens: list of tokens
        :return: list of token ids
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return input_ids
    
    def convert_ids_to_tokens(self, input_ids):
        """ Convert input ids to tokens
        
        :param input_ids: token ids
        :return: list of tokens
        """
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        return tokens

    def get_attention_masks(self, input_ids):
        """ Generate attention masks from list of input ids
        
        :param input_ids: list of input ids
        :return: attention masks
        """
        attention_masks = []
        for sentence in input_ids:
            attention_mask = [float(id > 0) for id in sentence]
            attention_masks.append(attention_mask)
        return attention_masks

    def text_preprocessing(self, text):
        """ Pre-process textual content and generate input ids, attention masks
        
        :param text: text
        :return: input ids, attention masks
        """
        tokenized_text = self.tokenize_text(text)
        input_ids = self.convert_tokens_to_ids(tokenized_text)
        attention_masks = self.get_attention_masks([input_ids])
        input_ids = torch.tensor([input_ids])
        attention_masks = torch.tensor([attention_masks])
        return input_ids, attention_masks
