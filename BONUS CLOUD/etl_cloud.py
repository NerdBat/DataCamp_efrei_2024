import string
import re
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment.util import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import warnings
from bs4 import BeautifulSoup as bs
import pandas as pd
from time import sleep
import numpy as np
from random import randint
import boto3 
import time
from io import StringIO
import redshift_connector


#Connection AWS Athena
AWS_ACCES_KEY = "AKIA47CRU6EFD575BE6N"
AWS_SECRET_KEY = "SECRET_KEY" #rentrer la secret_key
AWS_REGION = "eu-north-1"
SCHEMA_NAME = "election_2024"
S3_STAGING_DIR = "s3://test-bucket-election/output/"
S3_BUCKET_NAME = "test-bucket-election"
S3_OUTPUT_DIRECTORY = "output"


bucket = 'projet-election'
csv_buffer = StringIO()

s3_resource = boto3.resource(
    's3',
    aws_access_key_id=AWS_ACCES_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

#WebScrapping
headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebkit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
url = "https://debatepolitics.com/forums/2024-us-presidential-election.227/?order=reply_count&direction=desc"
response = requests.get(url)

if response.status_code == 200:
    soup = bs(response.content, 'html.parser')
    hrefs = [a.get('href') for a in soup.find_all('a') if a.get('href') and '/threads/' in a.get('href')]
else:
    print(f"Erreur lors de la requête HTTP: {response.status_code}")
    
liste_output = []
def get_good_url(liste_input):
    liste_input = liste_input[3:]
    i = 0
    while i < len(liste_input):
        liste_output.append(liste_input[4])
        liste_input = liste_input[6:]     
        i=i+1 
    return liste_output

new_ref = get_good_url(hrefs)

pages = [] # On stock le numéro de la page de fin
liens = [] # On stock l'url du lien  a traiter 
len_p = len('page-')
for v in new_ref:
    find_page = v.find('page-')
    tmp = v[(find_page+len_p):]
    pages.append(int(tmp))
    v = v[:v.rfind('-')+1]
    liens.append(v) # On récupère le numéro de pages et on le stock dans la liste liens
len(pages)

new_ref = get_good_url(hrefs)

pages = []
liens = []
len_p = len('page-')
for v in new_ref:
    find_page = v.find('page-')
    tmp = v[(find_page+len_p):]
    pages.append(int(tmp))
    v = v[:v.rfind('-')+1]
    liens.append(v)


data_text = []
data_gender = []
data_politics = []
data_localisation = [] 
nb_forums = np.arange(1,len(pages), 1)
url = 'https://debatepolitics.com/'

for index in nb_forums:
    page = pages[index]
    lien = liens[index]
    
    for p in range(page,1, -1):
        page = requests.get(url+lien+str(p))
        soup = bs(page.text, 'html.parser')
        acc_data = soup.find_all('div', class_= 'message-inner')
        #sleep(randint(2,6))
    
        for data in acc_data:
            gender = data.find('dl', {'class' : 'pairs pairs--justified', 'data-field': 'gender'})
            politics = data.find('dl', {'class' : 'pairs pairs--justified', 'data-field': 'political_leaning'})
    
            localisation = data.find('a', {'class' : 'u-concealed'})
    
            text = data.find('div', {'class' : 'bbWrapper'})
            div_to_remove = text.find('div', class_ = 'bbCodeBlock-expandContent js-expandContent') 
            if div_to_remove:
                div_to_remove.decompose()
            if text is not None:
                if gender:
                    data_gender.append(gender.text)
                else:
                    data_gender.append(None)
                if politics:
                    data_politics.append(politics.text)
                else:
                    data_politics.append(None)
                if localisation:
                    data_localisation.append(localisation.text)
                else:
                    data_localisation.append(None)
                data_text.append(text.text)


dico = {
    "gender": data_gender,
    "politics": data_politics,
    "localisation": data_localisation,
    "text": data_text
}

df = pd.DataFrame(dico)

df2 = df.copy()

#Nettoyage
word1_truncate = 'said:\n\n\n'
word1_len = len(word1_truncate)
word2_truncate = 'Click to expand...\n\n'
word2_len = len(word2_truncate)

for index in df2.index:
    tmp = df2.at[index, 'text']
    tr1 = tmp.rfind(word1_truncate)
    if tr1 != -1:
        tmp = tmp[tr1+word1_len:]

    tr2 = tmp.rfind(word2_truncate)
    if tr2 != -1:
        tmp = tmp[tr2+word2_len:]

    tmp = tmp.lstrip('\n')
    tmp = tmp.rstrip('\n')
        
    df2.at[index,'text'] = tmp
    
word_truncate1 = '\nGender\n'
len_word1 = len(word_truncate1)

for index in df2.index:
    if df2.at[index, 'gender']:
        tmp = df2.at[index, 'gender']
        tmp = tmp[len_word1:]
        df2.at[index, 'gender'] = tmp


word_truncate2 = '\n'
len_word2 = len(word_truncate2)

for index in df2.index:
    if df2.at[index, 'gender']:
        tmp = df2.at[index, 'gender']
        tmp = tmp[:-1] 
        df2.at[index, 'gender'] = tmp


word_truncate3 = '\nPolitical Leaning\n'
len_word3 = len(word_truncate3)

for index in df2.index:
    if df2.at[index, 'politics']:
        tmp = df2.at[index, 'politics']
        tmp = tmp[len_word3:]
        df2.at[index, 'politics'] = tmp


word_truncate4 = '\n'
len_word4 = len(word_truncate4)

for index in df2.index:
    if df2.at[index, 'politics']:
        tmp = df2.at[index, 'politics']
        tmp = tmp[:-1]
        df2.at[index, 'politics'] = tmp
        
        
pd.set_option('display.max_rows', None)
us_states_map = {
    'AL': 'Alabama',
    'KY': 'Kentucky',
    'NC': 'North Carolina',
    'SC': 'South Carolina',
    'NY': 'New York',
    'Long Island NY': 'New York',
    'Nassau County, Long Island': 'New York',
    'Central NY': 'New York',
    'Western New York': 'New York',
    'Western New York State': 'New York',
    'New York City area': 'New York',
    'The Big Apple': 'New York',
    'Fort Drum, New York': 'New York',
    'IL—16': 'Illinois',
    'KCMO & 50K Feet Up 4Reagan4Perot4Obama': 'Missouri',
    'Springfield MO': 'Missouri',
    'The St. Louis Metro': 'Missouri',
    'Near Boise, ID': 'Idaho',
    'North Idaho': 'Idaho',
    'Houston Area, TX': 'Texas',
    'Houston, in the great state of Texas': 'Texas',
    'North Texas': 'Texas',
    'Uhland, Texas': 'Texas',
    'DFW': 'Texas',
    'Third Coast': 'Texas',
    'Tucson': 'Arizona',
    'Tucson, AZ': 'Arizona',
    'arizona': 'Arizona',
    'Sarasota Fla': 'Florida',
    'Flori-duh': 'Florida',
    'Florida The Armband State': 'Florida',
    'Tampa Bay area': 'Florida',
    'Cambridge, MA': 'Massachusetts',
    'Western Mass.': 'Massachusetts',
    'Near Seattle': 'Washington',
    'Outside Seattle': 'Washington',
    'Seattle WA': 'Washington',
    'Washington State': 'Washington',
    'SoCal': 'California',
    'Southern California': 'California',
    'The Bay': 'California',
    'SF Bay Area': 'California',
    'Los Angeles': 'California',
    'San Diego': 'California',
    'Mentor Ohio': 'Ohio',
    'NE Ohio': 'Ohio',
    'Ohio, USA': 'Ohio',
    'Columbus, OH': 'Ohio',
    'Southern OR': 'Oregon',
    'Portlandia': 'Oregon',
    'Northern Nevada': 'Nevada',
    'SW Virginia': 'Virginia',
    'Northern New Jersey': 'New Jersey',
    'New Mexico, USA': 'New Mexico',
    'Tennessee, USA': 'Tennessee',
    'Bridgeport, CT': 'Connecticut',
    'Philadelphia': 'Pennsylvania',
    'Rolesville, NC': 'North Carolina',
    'Atlanta': 'Georgia'
}

# Dictionnaire pour les régions US
us_regions_map = {
    'Blue Ridge Mountains': 'Unknown',
    'Central Texas': 'Texas',
    'Greater Boston Area': 'Massachusetts',
    'Mid-West USA': 'Unknown',
    'N. Virginia': 'Virginia',
    'Western Virginia': 'Virginia',
    'North East': 'Unknown',
    'Pacific NW': 'Unknown',
    'PNW': 'Unknown',
    'US Southwest': 'Unknown',
    'PDX and ATL': 'Unknown',
    'New England, United States': 'Unknown'
}

# Dictionnaire pour les autres pays
countries_map = {
    'Best Coast Canada': 'Canada',
    'Lower Mainland of BC': 'Canada',
    'Vancouver, Canada Dual citizen': 'Canada',
    'North norfolk England': 'UK',
    'North norfolk  England': 'UK',
    'Southern England': 'UK',
    'Devonshire, England': 'UK',
    'Suierland, Germany': 'Germany',
    'Paris, France': 'France',
    'Tijuana, B.C., Mexico.': 'Mexico',
    '🇦🇹 Austria 🇦🇹': 'Austria',
    'U.S.': 'USA',
    'US': 'USA',
    'USofA': 'USA',
    'United States': 'USA',
    'United States, all over.': 'USA',
    'US of A': 'USA',
    'America, the place that has ALWAYS been great': 'USA'
}

def standardize_location(location):
    if pd.isna(location) or not str(location).strip():
        return 'Unknown'
    location = str(location)
    for mapping in [us_states_map, us_regions_map, countries_map]:
        if location in mapping:
            return mapping[location]
    return 'Unknown'

df2['localisation'] = df['localisation'].apply(standardize_location)

df2['id'] = 0
df2['Trump'] = False
df2['Kamala'] = False
df2['Democrat'] = False
df2['Republican'] = False

for index in df2.index:
    df2.at[index, 'id'] = tmp = index
    if 'trump' in df2.at[index,'text'].lower():
        df2.at[index, 'Trump'] = True
    if 'kamala' in df2.at[index,'text'].lower():
        df2.at[index, 'Kamala'] = True
    if 'democrat' in df2.at[index,'text'].lower():
        df2.at[index, 'Democrat'] = True
    if 'republic' in df2.at[index,'text'].lower():
        df2.at[index, 'Republican'] = True
df2.head()

chemin = "../Code/data_pour_modele.csv"
df2.to_csv(chemin, index=False)

nltk.download('averaged_perceptron_tagger') 
nltk.download('maxent_ne_chunker')  
nltk.download('words')  
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta_debog(example):
    try:
        encoded_text = tokenizer(example, return_tensors='pt')

        # Model prediction
        output = model(**encoded_text)

        # Extracting and processing scores
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
        return scores_dict

    except IndexError as e:
        return {'roberta_neg': None, 'roberta_neu': None, 'roberta_pos': None}
    
    
roberta = df2.copy()
from tqdm.notebook import tqdm


res = {}
for i, row in tqdm(roberta.iterrows(), total=len(roberta)):
    try:
        text = row['text']
        myid = row['id']
        roberta_result = polarity_scores_roberta_debog(text)
        res[myid] = roberta_result
    except RuntimeError:
        print(f'Broke for id {myid}')

df_copy_2 = df2.drop(index=599)

roserta_result = pd.DataFrame(res).T
roseta_output = roserta_result.reset_index().rename(columns={'index': 'Id'})
results_df = roserta_result.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(roberta, how='left', left_on='Id', right_on='id')


#insertion du csv dans output
results_df.to_csv(csv_buffer)
s3_resource.Object(bucket, 'output/infoCovid.csv').put(Body=csv_buffer.getvalue())

conn = redshift_connector.connect(
    host='redshift-cluster-2.ck9isbajzex0.eu-north-1.redshift.amazonaws.com',
    database='dev',
    port=5439,
    user='awsuser',
    password='mot_de_passe' 
)

conn.autocommit = True

cursor=redshift_connector.Cursor = conn.cursor()

# Supprime si elle existe
cursor.execute("""
DROP TABLE IF EXISTS forums
)
""")

#Création de la table 
cursor.execute("""
CREATE TABLE "forums" (
"id" INTEGER,
  "compound" FLOAT,
  "neu" FLOAT,
  "neg" FLOAT,
  "pos" FLOAT,
  "politics" TEXT,
  "text" TEXT,
  "localisation" TEXT,
  "gender" TEXT,
  "Trump" BOOLEAN,
  "Kamala" BOOLEAN,
  "Democrat" BOOLEAN,
  "Republican" BOOLEAN
)
""")

#Insertion dans la table
cursor.execute("""
copy date from 's3://projet-election/output/infoCovid.csv'
credentials 'aws_iam_role=arn:aws:iam::891376955658:role/s3-redshift-glue'
delimiter ','
region 'eu-north-1'
IGNOREHEADER 1
""")

