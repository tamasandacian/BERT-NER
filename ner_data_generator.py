import pandas as pd

class NerDataGenerator(object):
    """ Class for generating required data for an NER model

    :param df: DataFrame
    :param colname: column to group DataFrame 
    """
    def __init__(self, df, colname='group_by_colname'):
        
        self.df = df
        self.colname = colname

        agg_func = lambda s: [(w, p, t) for w, p, t in zip(
                                s["Word"].values.tolist(), 
                                s["POS"].values.tolist(), 
                                s["Tag"].values.tolist())]

        # group dataframe by group colname
        self.grouped = self.df.groupby(self.colname)
        self.grouped = self.grouped.apply(agg_func)
        """
        Sample:
            Sentence #
            Sentence: 1        [(Thousands, NNS, O), (of, IN, O), (demonstrat...
            Sentence: 10       [(Iranian, JJ, B-gpe), (officials, NNS, O), (s...
            Sentence: 100      [(Helicopter, NN, O), (gunships, NNS, O), (Sat...
            Sentence: 1000     [(They, PRP, O), (left, VBD, O), (after, IN, O...
            Sentence: 10000    [(U.N., NNP, B-geo), (relief, NN, O), (coordin...
                                                    ...                        
            Sentence: 9995     [(Opposition, NNP, O), (leader, NN, O), (Mir, ...
            Sentence: 9996     [(On, IN, O), (Thursday, NNP, B-tim), (,, ,, O...
            Sentence: 9997     [(Following, VBG, O), (Iran, NNP, B-geo), ('s,...
            Sentence: 9998     [(Since, IN, O), (then, RB, O), (,, ,, O), (au...
            Sentence: 9999     [(The, DT, O), (United, NNP, B-org), (Nations,...
        """

        # generate list of pairs from grouped colname
        self.pairs = [pair for pair in self.grouped]
        """
        Sample:
            [
                ('Thousands', 'NNS', 'O'),
                ('of', 'IN', 'O'),
                ('demonstrators', 'NNS', 'O'),
                ('have', 'VBP', 'O'),
                ('marched', 'VBN', 'O'),
                ('through', 'IN', 'O'),
                ('London', 'NNP', 'B-geo'),
                ('to', 'TO', 'O'),
                ('protest', 'VB', 'O'),
                ('the', 'DT', 'O'),
                ('war', 'NN', 'O'),
                ('in', 'IN', 'O'),
                ('Iraq', 'NNP', 'B-geo'),
                ('and', 'CC', 'O'),
                ('demand', 'VB', 'O'),
                ('the', 'DT', 'O'),
                ('withdrawal', 'NN', 'O'),
                ('of', 'IN', 'O'),
                ('British', 'JJ', 'B-gpe'),
                ('troops', 'NNS', 'O'),
                ('from', 'IN', 'O'),
                ('that', 'DT', 'O'),
                ('country', 'NN', 'O'),
                ('.', '.', 'O')
            ]
        """
        
        self.label_lists = [[item[2] for item in pair] for pair in self.pairs]
        """
        Sample:
           [
               'O', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 
               'O', 'B-gpe', 'O', 'O', 'O', 'O', 'O'
            ]

        """
       
        self.token_lists = [[item[0] for item in pair] for pair in self.pairs]
        """
        Sample:
            [
                'Thousands', 'of', 'demonstrators', 'have', 'marched', 'through', 'London', 'to', 'protest', 
                'the', 'war', 'in', 'Iraq', 'and', 'demand', 'the', 'withdrawal', 'of', 'British', 'troops', 
                'from', 'that', 'country', '.'
            ]
        """

        self.label_values = list(set([label for sublist in self.label_lists for label in sublist]))
        """
        Sample:
            [
                'O', 'I-art', 'I-per', 'B-nat', 'I-gpe', 'I-org', 'I-nat', 'B-per', 'I-tim', 'B-org', 'I-eve', 
                'B-eve', 'B-art', 'B-tim', 'B-gpe', 'I-geo', 'B-geo'
            ]
        """