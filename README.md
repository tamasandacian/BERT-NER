# BERT-NER

BERT-NER is an NLP task meant to help in identifying named entities in a given text. In this work we make use of existing open-source dataset to generate NER (Named Entity Recognition) model using state-of-the-art BERT pre-trained model.
The aformentioned dataset cand be found at the following url: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus#ner_dataset.csv

Technologies used: Python3, BERT, Jupyter Notebook, matplotlib, seaborn, Google Colab

## Project Outline:
```
  - Data Acquisition
  - Generating ground-truth label dataset
  - Model Training & Evaluation
  - Entity prediction

Basic project installation steps:

  1. Clone repository

  2. Generate model & evaluation files:
     - load dataframe                                 
     - check sequence length distribution in dataset  : required for BERT pre-trained model (max length 512 tokens)
     - use power of 2 to determine max_length           (e.g 128, 256, 512)
     - import and create Evaluation object
     - create model using create_model() function

          from evaluation import Evaluation
          
          df = pd.read_csv('data/ner_dataset.csv')
          df.fillna(method='ffill', inplace=True)
          df = df.rename(columns={'Sentence #': 'sentence_idx'})
          
          ev = Evaluation(
              lang_code='en', version="1.1", colname='sentence_idx', pre_trained_name='bert-base-cased', epochs=5
          )
          ev.create_model(df, max_length=128, output_path='output')

     Evaluation files:
        - data distribution: sequence length
        - plot train history
        - label_index json file   : label-index mapping
        
  3. Predict entities for new documents:
      - import and create NER object
      - predict entities using predict_entities() function
      
   Sample:
         
         from ner import NER
   
         text = """ Obama was born in Honolulu, Hawaii. After graduating from Columbia University in 1983, 
                    he worked as a community organizer in Chicago. In 1988, he enrolled in Harvard Law School, 
                    where he was the first black person to be president of the Harvard Law Review. After graduating, 
                    he became a civil rights attorney and an academic, teaching constitutional law at the University 
                    of Chicago Law School from 1992 to 2004. Turning to elective politics, he represented the 13th 
                    district from 1997 until 2004 in the Illinois Senate, when he ran for the U.S. Senate.  
                """
         
         ner = NER(lang_code="en", version="1.1", min_words=10, clean_text=False)
         pred = ner.predict_entities(text)
         print(pred)
         '''
             {
              'entities': [
                  {
                    'confidence': 0.9638853073120117,
                    'definition': 'Person',
                    'index': 0,
                    'label': 'B-per',
                    'token': 'Obama'
                  },
                  {
                    'confidence': 0.9939144253730774,
                    'definition': 'Geographical Entity',
                    'index': 4,
                    'label': 'B-geo',
                    'token': 'Honolulu'},
                  {
                    'confidence': 0.991294801235199,
                    'definition': 'Geographical Entity',
                    'index': 6,
                    'label': 'B-geo',
                    'token': 'Hawaii'
                  },
                  {
                    'confidence': 0.9797477126121521,
                    'definition': 'Organization',
                    'index': 11,
                    'label': 'B-org',
                    'token': 'Columbia'
                  },
                  {
                    'confidence': 0.983254611492157,
                    'definition': 'Organization',
                    'index': 12,
                    'label': 'I-org',
                    'token': 'University'
                  },
                  {
                    'confidence': 0.996993899345398,
                    'definition': 'Time Indicator',
                    'index': 14,
                    'label': 'B-tim',
                    'token': '1983'
                  },
                  {
                    'confidence': 0.9941704273223877,
                    'definition': 'Geographical Entity',
                    'index': 23,
                    'label': 'B-geo',
                    'token': 'Chicago'
                  },
                  {
                    'confidence': 0.9975076913833618,
                    'definition': 'Time Indicator',
                    'index': 26,
                    'label': 'B-tim',
                    'token': '1988'
                  },
                  {
                    'confidence': 0.9924579858779907,
                    'definition': 'Organization',
                    'index': 31,
                    'label': 'B-org',
                    'token': 'Harvard'
                  }
                  ....
               ],
              'message': 'successful'
            }
         '''
       
```
