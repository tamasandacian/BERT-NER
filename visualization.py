import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['figure.figsize'] = (10, 5)

def save_train_history(history, output_path):
    """ Save plot train history  
        
    :param history: dictionary
    :param output_path: output path
    """
    plt.figure()
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.title('Training history')
    plt.ylabel('Validation')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(output_path)
    plt.clf()

def save_seq_len_distribution(token_lists, output_path):
    """ Save plot text length distribution found in a dataset
        
    :param token_lists: documents as list of tokens
    :param output_path: output path
    """
    plt.figure()
    seq_len = [len(token_list) for token_list in token_lists]
    pd.Series(seq_len).hist(bins=30)
    plt.savefig(output_path)
    plt.clf()