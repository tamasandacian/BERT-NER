# BERT-NER

Is a project dedicated to Natural Language Processing (NLP) tasks, specifically Named Entity Recognition (NER), utilizing the BERT model. This project serves as a comprehensive exploration of performing named entity extraction on an email corpus. For this particular task, we employ the publicly available ‘Enron Email’ dataset, which can be accessed at the following URL: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset?datasetId=55.

In this project, we establish a baseline model using a combination of machine learning algorithms, including Perceptron, Passive Aggressive Classifier, Naive Bayes Classifier, and SGD Classifier. Subsequently, we compare the performance of these models with BERT for the NER task.

Furthermore, it is worth noting that this work has the potential for extension into the realm of multi-task learning problems. For instance, it could be expanded to encompass tasks such as NER and sentiment analysis.

The trained BERT NER models and data can be found at the following Google Drive URL: https://drive.google.com/drive/folders/1jPOMp7B6wirjEAwfsY9yVaihO9dqj-1C?usp=sharing


#### Project Outline:
```
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Data Anotation
4. Ground-truth creation
5. Model Training & Evaluation
6. Model Prediction
```

#### NER Tags
``` 
  * PERSON      : Person names e.g. first name, last name
  * ORG         : Companies, agencies, institutions, etc   
  * LOCATION    : Street address, city, country name
  * DATE        : Temporal expressions, relative dates and time periods
```

#### Setup
```
1. Clone repository

2. navigate to project repository
   cd BERT-NER

3. create virtual environment
   virtualenv -p python3 env
   source env/bin/activate

4. install required libraries
   pip3 install flairNLP
   pip3 install langchain
   pip3 install openai
   pip3 install scikit-learn
   pip3 install pandas
   pip3 install numpy

5. run command line
   jupyter notebook
```

#### 1.Exploratory Data Analysis

In this part, we start by performing data analysis to understand the email data we are dealing with. The analysis consists of (1) checking the email format, (2) understanding email distribution over time, (3) checking email subjects type, (4) the number of words in each email etc.

The email dataset consists of email records between 1979 - 2020 years where most emails are sent in 2001. I found that the dataset presents some mistakes where some records are from 2043 and 2044 year. This is a mistake in the given dataset as data is presented in future.

After analyzing the distribution of emails by month, the dataset consists of emails that were mainly sent in November month and in second place October month whereas less number of emails were sent in July. Regarding the distribution of emails by day, the dataset consists emails that were sent mostly in day 26 and 27 which represents the end of the month and emails that were sent less in the beggining of the month e.g. day 3.

There are many types of email formats found in the dataset such as: personal emails, discussion threads, sent emails, email found in inbox. Some of the emails come in short format as only few words or one word and long email format as replied / forwarded emails.

To perform NER on the email dataset, we start by doing a preliminary research on the internet to find existing annotation dataset for the Enron corpus.
By utilizing an annotated dataset, we can extract valuable annotated entities and merge them with the orginal dataset.

I found a short list of interesting resources that are mentioned in the paper "Annotating Large Email Datasets for Named Entity Recognition with Mechanical Turk".
The paper can be seen at the following url: https://faculty.washington.edu/melihay/publications/NAACL2010a.pdf

Resources:
- http://tides.umiacs.umd.edu/webtrec/trecent/parsed_w3c_corpus.html
- http://www.cs.cmu.edu/~wcohen/repository.tgz
- http://www.cs.cmu.edu/~einat/datasets.html

Out of this short list, it is worth noting that out of the mentioned resources, only one class PERSON was provided as for the LOCATION and ORG entity there is no available link to the annotation dataset to download.

For this reason, I decided to annotate a small subset of emails with OpenAI's GPT-3 model for training the NER model.

#### 2.Data Preprocessing

In this part, we start by performing preprocessing on the textual content. It consists of the following steps: (1) extract the last message from replied and forwarded emails, (2) cleaning punctuation marks, (3) extracting text from HTML content and (4) removing duplicate emails.

I found that most of replied emails have a defined structure. For example some forwarded emails include string such as "-------Forwarded by" in the email content or some replied emails have "-----Original Message" string included. We can use regex to match this pattern and disregard everything that is after. 
Here we only consider the top part of the email as it represents the last message received.

There are emails which don't contain this pattern but have other indicators for example message sent by someone on a specific date at a given time period "Phelim Boyle <pboyle@uwaterloo.ca> on 03/28/2000 06:23:07 PM". Other patterns include "From:", "To:" "LOG MESSAGES:", and mentioned date.

While working on extracting the content, I found that some emails contain html content. We can extract the textual content from it using BeautifulSoup library.
After processing the emails, I have seen that there are emails with only one word as replied message can be short e.g. "FYI", "agreed", "thanks", "great" etc, whereas other emails could have longer texts.

Once extracted the text, we will consider a small subset e.g. 1000 emails that have 100 to 150 words. The reason for this is that we want to extract meaningful entities using GPT-3 but at the same time avoid cost.
We can do this by giving instructions to GPT-3 model.

#### 3.Data Annotation

To perform data annotation, we can use OpenAI's GPT-3 model by providing few annotated email samples and instruct the model to perform named entity recognition as well as determine the overall sentiment.
The following presents few examples that are passed to the model. Here we are asking the model to perform few-shot learning and at the end to output the prediction as a Python dictionary.

```
examples = [
    {
        "query": """I don't know if you know the history behind McCullough.  He is a consultant 
                who primarily advises industrial loads.  He advised his customers that prices 
                were going down and that they should go on index based tariffs rather than 
                fixed-price tariffs.  Needless to say, his advise cost his customers a great 
                deal of money.  Notable examples are Bellingham Cold Storage and Georgia 
                Pacific in Puget Sound Energy's service territory.  Rather than acknowledge 
                that he was wrong, Robert is on a witch hunt.  His data is bad, his analysis 
                is horrible, he lies with numbers but does a good job being pompous and 
                putting together flashy power point presentations.  I have not heard of 
                anybody agreeing with him, including the NWPPC, CEC, CAISO, CAPX.  He is a 
                loud entertaining speaker, so I am sure that he will continue to get coverage.
        """,
        "answer": """
            "persons": ["McCullough"],
            "dates": [],
            "locations": [],
            "organizations": ["Bellingham Cold Storage", "Georgia Pacific", "Puget Sound Energy's"],
            "sentiment": "negative"
        """
    },
    {
        "query": """Reagan,
    
                I sent you an email last week stating that I would be in San Marcos on 
                Friday, April 13th.  However, my closing has been postponed.  As I mentioned 
                I am going to have Cary Kipp draw the plans for the residence and I will get 
                back in touch with you once he is finished.

                Regarding the multifamily project, I am going to work with a project manager 
                from San Antonio.  For my first development project, I feel more comfortable 
                with their experience obtaining FHA financing.  We are working with Kipp 
                Flores to finalize the floor plans and begin construction drawings.  Your bid 
                for the construction is competive with other construction estimates.  I am 
                still attracted to your firm as the possible builder due to your strong local 
                relationships.  I will get back in touch with you once we have made the final 
                determination on unit mix and site plan.

                Phillip Allen""",
          "answer": """
              "persons": ["Reagan", "Cary Kipp", "Kipp Flores", "Phillip Allen"],
              "dates": ["last week", "Friday, April 13th"],
              "locations": ["San Marcos", "San Antonio"],
              "organizations": [],
              "sentiment": "neutral"
          """
      },
      {
          "query": """Thanks for the response.  I think you are right that engaging an architect is 
                  the next logical step.  I had already contacted Cary Kipp and sent him the 
                  floor plan.
                  
                  He got back to me yesterday with his first draft.  He took my plan and 
                  improved it.  I am going to officially engage Cary to draw the plans.  While 
                  he works on those I wanted to try and work out a detailed specification list.  
                  Also, I would like to visit a couple of homes that you have built and speak to 
                  1 or 2 satisfied home owners.  I will be in San Marcos on Friday April 13th.  
                  Are there any homes near completion that I could walk through that day?  Also can 
                  you provide some references?
  
                  Once I have the plans and specs, I will send them to you so you can adjust 
                  your bid.  
  
                  Phillip""",
          "answer": """
              "persons": ["Cary Kipp", "Cary", "Phillip"],
              "dates": ["yesterday", "on Friday April 13th"],
              "locations": ["San Marcos"],
              "organizations": [],
              "sentiment": "positive"
          """
      }
]
```

```python
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.chat_models import ChatOpenAI
from langchain import FewShotPromptTemplate
from langchain import PromptTemplate
from langchain import LLMChain

# Initialize llm object
openai_api_key = "REPLACE_WITH_YOUR_OPENAI_API_KEY"

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=2048
)

# create a example template
example_template = """
Email: {query}
Output: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """

You are an intelligent assistant.
For the following email text, extract the following information:

persons: Extract person names and output them as a comma separated Python list.

locations: Extract location names such as street address, city, country \
and output them as a comma separated Python list.

organizations: Extract company names and output them as a comma separated Python list.

dates: Extract dates found such as temporal expressions, relative dates and time periods
and output them as a comma separated Python list.

sentiment: Determine the overal sentiment in text as positive, negative or neutral \
and output as a Python string.

Format the output as Python dictionary with the following keys:
persons
locations
organizations
dates
sentiment
"""

# and the suffix our user input and output indicator
suffix = """
Email: {query}
Output: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=1500  # this sets the max length that examples should be
)

# now create the few shot prompt template
dynamic_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,  # use example_selector instead of examples
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)

query = """Bob,

Patti Sullivan held together the scheduling group for two months while Randy 
Gay was on a personal leave.  She displayed a tremendous amount of commitment 
to the west desk during that time.  She frequently came to work before 4 AM 
to prepare operations reports.  Patti worked 7 days a week during this time.  
If long hours were not enough, there was a pipeline explosion during this 
time which put extra volatility into the market and extra pressure on Patti.  
She didn't crack and provided much needed info during this time.

 Patti is performing the duties of a manager but being paid as a sr. 
specialist.  Based on her heroic efforts, she deserves a PBR.  Let me know 
what is an acceptable cash amount.

Phillip
"""

llm_chain = LLMChain(
    prompt=dynamic_prompt_template,
    llm=llm
)

result = llm_chain.run(query)
'''
{'persons': ['Patti Sullivan', 'Randy Gay', 'Phillip'], 'dates': [], 'locations': [], 'organizations': [], 'sentiment': 'positive'}
'''
```

After labeling our dataset we end up with tokens labeled as PERSON 2337 times, ORG 1411 times, DATE 874 times and LOCATION 750 times.
There were few instances where the model could not enclose the prediction output with curly braces "{ .. }". This was an easy fix as I could enclose the prediction within curly braces as a post-processing step.

<img width="1039" alt="image" src="https://github.com/tamasandacian/BERT-NER/assets/11573356/09b7f044-766d-4473-a22c-a1853d5cdcb9">


#### 4.Ground-truth Creation

To generate ground-truth dataset for training a model an NER model, it is required that we label all text tokens into IOB (Inside-Outside-Beginning) format.

We will treat this as token classification problem where we classify each token into one of the given labels PERSON, LOCATION, ORG, and DATE.

In order to do this, we first convert our labels into IOB format and second we match the labeled tokens in text.

Our tokens will be labeled with the following tags: B-PERSON, I-PERSON, B-LOCATION, I-LOCATION, B-ORG, I-ORG, B-DATE, I-DATE, and O which means others.

```python

sample_email = """
Dale:

Looks okay  I just made a few changes  I updated the reference the NYISO 
as the site publisher and also updated the location reference.  

ISO NY Zone A (West) Peak

The Floating Price during a Determination Period shall be the average of the 
hourly day-ahead prices listed in the Index (in final, not estimate, form) 
for electricity delivered during Peak hours on each Delivery Day during the 
applicable Determination Period.  The Floating Price for each Determination 
Period shall be calculated utilizing the hourly clearing prices published by 
the New York Independent System Operator on its official web site currently 
located at http://www.nyiso.com/oasis/index.html, or any successor thereto, 
under the headings "Day Ahead Market LBMP - Zonal; Zonal Prices; West 
(61752)"  (the "Index").
"""

# Labeled email tokens with IOB format:
token_label = {
'Dale': 'B-PERSON', ':': 'O', 'Looks': 'O', 'okay': 'O', 'I': 'O', 'just': 'O',
'made': 'O','a': 'O', 'few': 'O', 'changes': 'O', 'updated': 'O', 'the': 'O',
'reference': 'O', 'NYISO': 'B-LOCATION', 'as': 'O', 'site': 'O', 'publisher': 'O',
'and': 'O', 'also': 'O', 'location': 'O', 'reference.': 'O', 'ISO': 'O', 'NY': 'O',
'Zone': 'O', 'A': 'O', '(West': 'O', ')': 'O', 'Peak': 'O', 'The': 'O', 'Floating': 'O',
'Price': 'O', 'during': 'O', 'Determination': 'O', 'Period': 'O', 'shall': 'O', 'be': 'O',
'average': 'O', 'of': 'O', 'hourly': 'O', 'day-ahead': 'O', 'prices': 'O', 'listed': 'O',
'in': 'O', 'Index': 'O', '(in': 'O', 'final': 'O', ',': 'O', 'not': 'O', 'estimate': 'O',
'form': 'O', 'for': 'O', 'electricity': 'O', 'delivered': 'O', 'hours': 'O', 'on': 'O',
'each': 'O', 'Delivery': 'O', 'Day': 'O', 'applicable': 'O', 'Period.': 'O', 'calculated': 'O',
 'utilizing': 'O', 'clearing': 'O', 'published': 'O', 'by': 'O', 'New': 'O', 'York': 'O',
'Independent': 'O', 'System': 'O', 'Operator': 'O', 'its': 'O', 'official': 'O', 'web': 'O',
'currently': 'O', 'located': 'O', 'at': 'O', 'http': 'O', '/': 'O', '/www.nyiso.com': 'O',
'/oasis': 'O', '/index.html': 'O', 'or': 'O', 'any': 'O', 'successor': 'O', 'thereto': 'O',
'under': 'O', 'headings': 'O', '"Day': 'O', 'Ahead': 'O', 'Market': 'O', 'LBMP': 'O', '-': 'O',
'Zonal': 'O', ';': 'O', 'Prices': 'O', 'West': 'O', '(61752': 'O', ')"': 'O',
'(the': 'O', '"Index"': 'O', ').': 'O', 'Leslie': 'B-PERSON'
}
```

Once converted the dataset into required format, we split the dataset into train 80%, and 20% validation and test sets.
The datasets are saved as .txt files where each line represents the token from email text and its associated IOB tag.

#### IOB Label Data Distribution

The following presents IOB label data distribution for train, validation and test set.

We did not include the "O" token as this would not show the labeled tokens as PERSON, LOCATION, ORG, and DATE in the chart
representation.

<img width="1020" alt="image" src="https://github.com/tamasandacian/BERT-NER/assets/11573356/2eb1588c-5b5e-49dd-ab13-69b21c5b89fe">

#### 5. Model Training & Model Evaluation

##### Determining the baseline model

For our baseline model we use a combination of machine learning algorithms, including Perceptron, Passive Aggressive Classifier, Naive Bayes Classifier, and SGD Classifier. We compare the results using F1-score metric.

<img width="1059" alt="image" src="https://github.com/tamasandacian/BERT-NER/assets/11573356/1470e53a-60de-4193-ac03-18cd6f8859a7">

<img width="1036" alt="image" src="https://github.com/tamasandacian/BERT-NER/assets/11573356/7610bb8e-fc5e-4bf2-934b-6af62458577e">

Based on our results, the best model with F1-score is Passive Aggresive Classifier. This will be used to compare with the trained BERT model for NER.

###### BERT

Training an NER model using flairNLP is easy as we only need to provide the path to the generated train, validation and test .txt files.
The trained model achieved a performance of 0.60% macro average F1-score. This was achieved after training the model for 10 iterations using recommended hyperaparameters such as: learning rate and batch size.

After comparing the results with the baseline model we can see that BERT achieved higher performance than the baseline model.

<img width="1037" alt="image" src="https://github.com/tamasandacian/BERT-NER/assets/11573356/e1b537e6-24f2-4a87-b42d-40c989c108be">

#### 5. Model Prediction

```python
from flair.models import SequenceTagger
from flair.data import Sentence

# load the trained model with best F1-score
model = SequenceTagger.load('models/taggers/ner/202309242015/final-model.pt')

# Create utility function to predict entities in text

def predict(text):
    """
    Function to predict entities
    """
    sentence = Sentence(text)

    # predict tags and print
    model.predict(sentence)

    entities = []
    for label in sentence.get_labels():
        entity = dict()
        entity["word"] = label.data_point.text
        entity["label"] = label.value
        entity["confidence"] = float("{0:.5f}".format(label.score))
        entities.append(entity)

    return entities

##############################################################################
sample_email = """
Tina,

Koch never returned my calls. Based on the contractual information, we may be 
putting ourselves in a risk area by 'assigning' the deals without proper 
documentation. The documentation that was supplied did not refence any ENA 
agreements, nor any specific meter numbers. If the purchase was indeed a 
meter/well sale; there should be documentation listing the meters acquired 
and the effective date. Since we are apparently the supplier, the risk is not 
as substaintial as if we were the purchaser. However, without documentation 
we are basically relying on the relationship btw Koch & Duke.

Cyndie
ENA Global Contracts

Tina Valadez@ECT
"""
predict(sample_email)
[{'word': 'Tina', 'label': 'PERSON', 'confidence': 0.95944},
 {'word': 'Koch', 'label': 'PERSON', 'confidence': 0.73476},
 {'word': 'ENA', 'label': 'ORG', 'confidence': 0.92136},
 {'word': 'Cyndie', 'label': 'PERSON', 'confidence': 0.61575},
 {'word': 'ENA Global', 'label': 'ORG', 'confidence': 0.80697},
 {'word': 'Tina Valadez', 'label': 'PERSON', 'confidence': 0.99369},
 {'word': 'ECT', 'label': 'ORG', 'confidence': 0.96524}]

##############################################################################
sample_email = """
Vince,

  I agree with you that it's a lesson people need to learn
over and over again.  I can't tell you how many politicians
I met over the past year who really don't like markets and
certainly don't understand how or why they work.  These
aren't just the minor leaguers in Sacramento, but the big
league players in Washington.

  I also agree that the academic community can play "an important
role in shaping public opinion and in explaining the logic of
deregulation process."  I'd like to think that is in large
part what I have been trying to do.

Frank
"""
predict(sample_email)
[{'word': 'Vince', 'label': 'PERSON', 'confidence': 0.99705},
 {'word': 'Sacramento', 'label': 'LOCATION', 'confidence': 0.98591},
 {'word': 'Washington', 'label': 'LOCATION', 'confidence': 0.8984},
 {'word': 'Frank', 'label': 'PERSON', 'confidence': 0.9975}]

############################################################################

sample_email = """
I wanted to follow-up with everyone following yesterday's meeting. 

It appears to me that we need to develop (1) a better analysis of the four 
market models - AGL (Steve M), Columbia of Ohio (Janine), Socal Gas (Jeff), 
NICOR (Roy) - based on some key elements and (2) the key Influence parties in 
Illinois with 5 layer Influence Circles that we need to be thinking about in 
this discussion.

It would be great if we could get this to Laura for distribution by end of 
day Friday (realize it's tight).

Key elements for Market Models 

1. Direct Access Allowed?  For What customers?  What timeline?
2. Upstream Capacity resolution - assignment or otherwise?  Are there assets 
left to optimize
3. Retail pricing - Fixed price vs. PBR (how?) vs. something else?
4. Role of Wholesale Outsourcing Agent (would ENA sell to marketers or 
utilities or both?)
5. Other Issues

Thanks,

Jim
"""
predict(sample_email)
[{'word': 'yesterday', 'label': 'DATE', 'confidence': 0.70943},
 {'word': 'Steve', 'label': 'PERSON', 'confidence': 0.61155},
 {'word': 'of', 'label': 'ORG', 'confidence': 0.85975},
 {'word': 'Janine', 'label': 'PERSON', 'confidence': 0.95581},
 {'word': 'Socal Gas', 'label': 'ORG', 'confidence': 0.80789},
 {'word': 'Jeff', 'label': 'PERSON', 'confidence': 0.99384},
 {'word': 'NICOR', 'label': 'PERSON', 'confidence': 0.96538},
 {'word': 'Roy', 'label': 'PERSON', 'confidence': 0.99177},
 {'word': 'Illinois', 'label': 'LOCATION', 'confidence': 0.97919},
 {'word': 'Laura', 'label': 'PERSON', 'confidence': 0.99627},
 {'word': 'end', 'label': 'DATE', 'confidence': 0.62944},
 {'word': 'Friday', 'label': 'DATE', 'confidence': 0.64875},
 {'word': 'ENA', 'label': 'ORG', 'confidence': 0.88702},
 {'word': 'Jim', 'label': 'PERSON', 'confidence': 0.99733}]
```

