from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from normalization import TextNormalization
from ner_data_generator import NerDataGenerator
import torch

class Dataset(object):
    """ Class for pre-processing Dataset and generate train, test datasets
    """
    def __init__(self, tokenizer, colname='group_by_colname', test_size=0.10, random_state=42, batch_size=32):
        
        self.tokenizer = tokenizer
        self.colname = colname
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.tn = TextNormalization(tokenizer)

    def preprocessing(self, df, max_length=128):
        """ Pre-process dataframe and generate train, test data

        :param df: DataFrame
        """
        # process and generate tag_values, sentences, labels from dataframe
        ndg = NerDataGenerator(df=df, colname=self.colname)
        
        # tokenize text tokens and preserve labels
        token_label_pairs = [
                self.tn.tokenize_pair(tokens, labels) 
                      for tokens, labels in zip(ndg.token_lists, ndg.label_lists)
        ]
        token_lists = [token_label_pair[0] for token_label_pair in token_label_pairs]
        label_lists = [token_label_pair[1] for token_label_pair in token_label_pairs]
        
        # get generated label values
        label_values = ndg.label_values
        # add new element to padd token labels to max_length
        label_values.append("PAD")
        
        # assign index position to each label value and convert it to numeric format
        label_index = {label: index for index, label in enumerate(label_values)}
        label_ids = [[label_index.get(l) for l in labels] for labels in label_lists]
    
        # convert text tokens to BERT token ids
        input_ids = [self.tn.convert_tokens_to_ids(tokens) for tokens in token_lists]

        # pad sequences to max_length
        input_ids = self.apply_padding(input_ids, value=0.0, max_length=max_length)
        label_ids = self.apply_padding(label_ids, value=label_index["PAD"], max_length=max_length)
        
        # get attention masks
        attention_masks = self.tn.get_attention_masks(input_ids)

        # split into train, test sets
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(
            input_ids, label_ids, random_state=self.random_state, test_size=self.test_size)
        train_masks, val_masks, _, _ = train_test_split(
            attention_masks, input_ids, random_state=self.random_state, test_size=self.test_size)

        # convert to torch tensors
        train_inputs = torch.tensor(train_inputs)
        val_inputs = torch.tensor(val_inputs)
        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)
        train_masks = torch.tensor(train_masks)
        val_masks = torch.tensor(val_masks)

        # generate train, validation dataloader
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=self.batch_size)

        self.ndg = ndg
        self.label_index = label_index
        self.n_classes = len(label_index)
        self.token_lists = token_lists
        self.label_lists = label_lists
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.label_ids = label_ids
        
        self.train_inputs = train_inputs
        self.val_inputs = val_inputs
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.train_masks = train_masks
        self.val_masks = val_masks
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def apply_padding(self, sequences, value, max_length):
        """ Apply padding to fix sentences with different length
        
        :param sequences:
        :param value: 
        :return: sequences
        """
        sequences = pad_sequences(
            sequences, 
            maxlen=max_length, 
            dtype="long", 
            value=value, 
            truncating="post", 
            padding="post"
        )
        return sequences

