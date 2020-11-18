import transformers
from transformers import BertForTokenClassification, BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
from visualization import save_train_history, save_seq_len_distribution
from shared.utils import dump_to_pickle
from shared.utils import dump_to_json
from shared.utils import dump_to_txt
from shared.utils import make_dirs
from collections import defaultdict
from dataset import Dataset
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluation(object):
    """ Class for generating NER model and evaluation files.
    
    :param lang_code: language model
    :param version: model version number
    :param pre_trained_name: BERT pre-trained model name
    :param epochs: number of times for learning the model
    :param batch_size: num samples for model to learn at a time in an epoch
    :param lr: learning rate
    :param eps: epsilon parameter used for stability learning the model
    :param colname: process dataframe group by colname and learn data from
    :param test_size: the size of test dataset
    :param random_state: param used for reproducibility results
    """
    def __init__(self, lang_code, version="1.1", pre_trained_name='bert-base-uncased', epochs=4, batch_size=32, 
                 lr=3e-5, eps=1e-8, colname='group_by_colname', test_size=0.10, random_state=42):
        
        self.lang_code = lang_code
        self.version = version
        self.epochs = epochs
        self.lr = lr
        self.eps = eps
        self.colname = colname
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.pre_trained_name = pre_trained_name

    def create_model(self, df, max_length, output_path):
        """ Create & save model, train, validation files to a given output path

        :param df: DataFrame
        :param max_length: max input length for training model
        :param output_path: path to save model, evaluation files
        """

        # define output path
        subdir = "{}_{}".format(self.lang_code, self.version)
        label_index_path = output_path + "/label_index/" + subdir
        models_path = output_path + "/models/" + subdir
        eval_path = output_path + "/evaluation/" + subdir

        # create directories
        make_dirs(output_path)
        make_dirs(label_index_path)
        make_dirs(models_path)
        make_dirs(eval_path)
        
        # create BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained(self.pre_trained_name)

        # create Dataset object and process dataframe
        dataset = Dataset(
            tokenizer, colname=self.colname, test_size=self.test_size, 
            random_state=self.random_state, batch_size=self.batch_size
        )

        dataset.preprocessing(df, max_length=max_length)
        model = BertForTokenClassification.from_pretrained(
            self.pre_trained_name,
            num_labels=dataset.n_classes,
            output_attentions = False,
            output_hidden_states = False
        )
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=self.lr, eps=self.eps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataset.train_dataloader) * self.epochs
        )

        # set initial loss to infinite
        best_val_loss = float('inf')
        history = defaultdict(list)
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1} / {self.epochs}')
            print("-" * 10)

            train_loss = self.train(
                model=model,
                dataloader=dataset.train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_samples=len(dataset.train_inputs)
            )
            print(f'Train loss {train_loss}')

            val_loss = self.evaluate(
                model=model,
                dataloader=dataset.val_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_samples=len(dataset.val_inputs)
            )
            print(f'Val loss {val_loss}')
            print()
            

            # track history for train, val losses
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), models_path + '/model.pt')

            dump_to_pickle(dataset.train_inputs, eval_path + '/train_inputs.pkl')
            dump_to_pickle(dataset.train_masks, eval_path + '/train_masks.pkl')
            dump_to_pickle(dataset.train_labels, eval_path + '/train_labels.pkl')
            dump_to_pickle(dataset.val_inputs, eval_path + '/val_inputs.pkl')
            dump_to_pickle(dataset.val_masks, eval_path + '/val_masks.pkl')
            dump_to_pickle(dataset.val_labels, eval_path + '/val_labels.pkl')
            dump_to_json(dataset.label_index, label_index_path + '/label_index.json', sort_keys=False)
            save_train_history(history, eval_path + "/train_history.png")
            save_seq_len_distribution(dataset.ndg.token_lists, eval_path + '/seq_length.png')

    def train(self, model, dataloader, optimizer, scheduler, num_samples):
        """ Train model using train dataloader with total number of documents in train set 
        
        :param model: BERT pre-trained model
        :param dataloader: generated torch train dataloader
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param num_samples: total number of train samples
        :return: total correct predictions, average loss
        """
        model = model.train()
        total_loss = 0

        for step, batch in enumerate(dataloader):
            model.zero_grad()
            
            # add batch to cpu/gpu
            batch = [r.to(device) for r in batch]
            input_ids, attention_mask, labels = batch

            # forward pass
            outputs = model(input_ids, attention_mask, labels=labels)

            # get the loss
            loss = outputs[0]
            # perform backward pass to calculate the gradients
            loss.backward()
            
            # track train loss
            total_loss += loss.item()

            # clip the gradients to 1.0. This will prevent exploding gradients problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update parameters
            optimizer.step()
            # update the learning rate
            scheduler.step()

        train_loss = total_loss / len(dataloader)
        return train_loss

    def evaluate(self, model, dataloader, optimizer, scheduler, num_samples):
        """ Evaluate model using validation dataloader with total number of documents in validation set 
        
        :param model: BERT pre-trained model
        :param dataloader: generated torch train dataloader
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param num_samples: total number of train samples
        :return: total correct predictions, average loss
        """
        model = model.eval()
        total_loss = 0
        
        for step, batch in enumerate(dataloader):

            batch = [r.to(device) for r in batch]
            input_ids, attention_mask, labels = batch

            # telling the model not to compute or store gradients
            with torch.no_grad():
                # forward pass, calculate logit predictions
                outputs = model(input_ids, attention_mask, labels=labels)

            # get the loss
            loss = outputs[0]

            # track train loss
            total_loss += loss.item()
            
        val_loss = total_loss / len(dataloader)
        return val_loss


    