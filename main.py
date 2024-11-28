'''
Importation des differents modules et librairie pour le scrapping de données:
- Request pour faire des requetes HTTP
- BeautifulSoup pour le webscrapping et les requêtes sur les balises HTML,
- Pandas pour la transformation en dataframe, 
- Sleep et Randint sont utiliser pour contourner certaine protection de site web, par exemple, si l'on envoie trop de requêtes a un site web en trop peu de temps, il peut nous bloquer l'accès a son site,
on utilise donc la fonction sleep pour "simuler" du comportement humain.
- Numpy pour l'utilisations d'array pour ranger les pages que l'on va scrapper.
'''

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
from time import sleep

from random import randint

# Mise en place du header pour le  webscraping
headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebkit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
url = "https://debatepolitics.com/forums/2024-us-presidential-election.227/?order=reply_count&direction=desc"

response = requests.get(url)

if response.status_code == 200:
    # Verifier la validité de la requête 
    # Parser le contenu HTML avec BeautifulSoup
    soup = bs(response.content, 'html.parser')

    # Trouver toutes les balises <a> et extraire les valeurs href qui contiennent "/threads/"
    hrefs = [a.get('href') for a in soup.find_all('a') if a.get('href') and '/threads/' in a.get('href')]

    # Afficher la liste des href filtrées
    #print(hrefs)
else:
    print(f"Erreur lors de la requête HTTP: {response.status_code}")

liste_output = []

def get_good_url(liste_input):
    # On retire les 3 première lignes, car il s'agit de lien qui ne concerne pas la politique
    liste_input = liste_input[3:]
    i = 0
    while i < len(liste_input):
        # On ne récupère que la derniere page de chaque forum et on supprime les autres liens
        liste_output.append(liste_input[4])
        liste_input = liste_input[6:]     
        i=i+1 
    return liste_output

new_ref = get_good_url(hrefs)

new_ref

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

data_text = []
data_gender = []
data_politics = []
data_localisation = [] 
# Création de liste pour le webscrapping
nb_forums = np.arange(0,len(pages), 1)
url = 'https://debatepolitics.com/' # URL de base

# Boucle qui sert a parcourir une page parmis notre liste de pages et aller recuperer les informations qui nous interesse. ( contenu du texte, le genre, le bord politique, la localisation)
for index in nb_forums:
    page = pages[index]
    lien = liens[index]
    
    for p in range(page,1, -1): # Décrémentation pour parcourir les pages dans l'ordre inverse
        page = requests.get(url+lien+str(p))
        soup = bs(page.text, 'html.parser')
        acc_data = soup.find_all('div', class_= 'message-inner')
        #sleep(randint(2,6))
    
        for data in acc_data:
            #Récupération du genre, duu bord politique, texte et localisation
            gender = data.find('dl', {'class' : 'pairs pairs--justified', 'data-field': 'gender'})
            politics = data.find('dl', {'class' : 'pairs pairs--justified', 'data-field': 'political_leaning'})
    
            localisation = data.find('a', {'class' : 'u-concealed'})
    
            text = data.find('div', {'class' : 'bbWrapper'})
            # supprime les réponses des utilisateur pour pas avoir de doublons
            div_to_remove = text.find('div', class_ = 'bbCodeBlock-expandContent js-expandContent') 
            if div_to_remove:
                div_to_remove.decompose()
            # Début de Nettoyage du code
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


#création du dictionnaire pour pouvoir ranger nos données
dico = {
    "gender": data_gender,
    "politics": data_politics,
    "localisation": data_localisation,
    "text": data_text
}


# Conversion de notre dictionnaire en dataframe pour le traitement des donnée
df = pd.DataFrame(dico)
len(df)

# On travail sur une copie pour eviter la redondansce
df2 = df.copy()

# Exploration du dataframe

# Nettoyage de la colonnes texte
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

    
#N Nettoyage de la colonne Gender et Political
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
        tmp = tmp[:-1] #points de l'autre côté pour supprimer la fin
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

# Nettoyage et transformation de la colonnes localisation
pd.set_option('display.max_rows', None)
# Dictionnaire pour les états US
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
    """
    Standardise une localisation en utilisant les dictionnaires de mapping.
    Si la localisation n'est pas trouvée ou est vide/NaN, renvoie 'Unknown'.
    """
    import pandas as pd
    
    # Gestion des valeurs NaN et vides
    if pd.isna(location) or not str(location).strip():
        return 'Unknown'
    
    # Conversion en string pour être sûr
    location = str(location)
        
    # Vérifier dans l'ordre: états US, régions US, pays
    for mapping in [us_states_map, us_regions_map, countries_map]:
        if location in mapping:
            return mapping[location]
            
    return 'Unknown'

# Nettoyage de la colonne localisation
df2['localisation'] = df['localisation'].apply(standardize_location)


# Ajout de colonnes indicative pour les mots cléf liée à l'éléction
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


# Exportation du dataframe finale apres nettoyage, pret pour le preprocessing
chemin = "../Code/data_pour_modele.csv"
df2.to_csv(chemin, index=False)


# ANALYSE 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore
import plotly.express as px

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
from tqdm.notebook import tqdm
from collections import Counter
from wordcloud import WordCloud
import nltk

# Télécharger les ressources nécessaires pour NER
nltk.download('averaged_perceptron_tagger') 
nltk.download('maxent_ne_chunker')  
nltk.download('words')  
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
import warnings
warnings.filterwarnings('ignore')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

texte = "I think trump is a fool"
sia.polarity_scores(texte)

res = {}
for i, row in tqdm(df2.iterrows(), total=len(df2)):
    text = row['text']
    myid = row['id']
    
    # Vérifier si 'text' est une chaîne de caractères ou le convertir en chaîne
    if isinstance(text, str):  # Si le texte est déjà une chaîne
        res[myid] = sia.polarity_scores(text)
    else:
        # Si le texte est de type 'float' (par exemple NaN ou nombre), on le convertit en chaîne vide
        res[myid] = sia.polarity_scores(str(text))  # Convertir en chaîne avant d'utiliser

# Utilisation de vaders pour les commentaires
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})

# Jointure entre nos DataFrame pour pouvoir faire des graphiques et avoir des infos
vaders = vaders.merge(df2, how='left', left_on='Id', right_on='id')

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
        # Tokenization
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

roberta_df = pd.DataFrame(res).T
roberta_output = roberta_df.reset_index().rename(columns={'index': 'Id'})
results_df = roberta_df.reset_index().rename(columns={'index': 'Id'})


results_df = results_df.merge(vaders, how='left', left_on='Id', right_on='id')
results_df.columns
'''
nltk = "../Code/nltk_result.csv"
vaders.to_csv(nltk, index=False)
rosa = "../Code/rosa_result.csv"
roserta_output.to_csv(rosa, index=True)
'''
print("pour récupérer le fichier de donnée il faut spécifier le chemin ou vous voulez exporter le fichier on veux un format comme cela '../Code/commun_result.csv' IL est primordiale de bien mettre le nom de fichier.csv pour bien executer le code")
#commun = input("Chemin de destination avec le nom du fichier.csv")
results_df.to_csv(index=False)
