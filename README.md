# Data Camp Groupe 4


Projet DataCamp du groupe 4.
- Rick Georges YELEUMEU
- François NGY
- Lauryn LESEL
- Antoine THEISSEN
- Hadjara KAKA

## ElectoPulse

Bienvenue dans le ReadME de nôtre projet DataCamp.

Pour commencer il faut commencer par récuperer les fichiers et installer toutes les dépendances, mais pour vous le travail sera facile.

On télécharge le Dépot Github dans un dossier de nôtre choix.
```bash
git clone https://github.com/NerdBat/DataCamp_efrei_2024
```
Ici on va installer toutes les dépendance
```bash
pip install -r requirements.txt
```
Enfin On va pouvoir lancer le code. vous pouvez executer le fichier main.ipynb dans [Visual Studio Code]('https://code.visualstudio.com/'), [Jupyter Notebook]('https://jupyter.org/') ou vôtre éditeur de code.

Penser a bien lancer la commande pour le "requirement.txt" afin de n'avoir aucun problème lors de l'execution.

Une fois le code executer vous aurez un fichier nommée data.csv.

Pour la visualisation il vous faudra utiliser PowerBi et avoir la version de Powerbi 2.138 ( vous pouvez la télécharger sur le Microsoft Store )


# Comment ça fonctionne ?

## Le scrapping de donnée : 

La première partie du code est dédiée au webscrapping, en effet pour nôtre problématique il n'existe pas de dataset déjà fait près a l'emploi sur lequel on n'aurais qu'a faire de l'analyse de sentiment et de la viusalisation a partir de nos résultats. On a donc du faire nôtre propre jeux de données. On a scrapper les donnée de [Politic Debates]('https://debatepolitics.com/'), plus spécifiquement la liste des threads lié a l'éléction 2024. On a récupéré le lien de tous les liens liée a ce sujets, inspéctée chaque pages une par une en récupérant le texte, et les informations de l'utilisateur qui a écris le commentaires on les a stocker dans un DataFrame Pandas avant de passer en phase de nettoyage de donnée. 

## Le nettoyage de donnée : 

Le nettoyage de donnée est une partie primordial dans le traitement des données, ici on a commencer par supprimer les réponses, car cela crée des doublons de texte, \n sont supprimer, les donnée renseigner dans "gender" et "politics" sont normalisé pour pouvoir être utilisé.
La partie localisation elle est soumis à de la transformation du a des informations qui sont soient erroner soit mal renseigner. Des listes de donnée propres sont utilisé pour remplacer certaine donnée non conformes.

## L'analyse de sentiment :

Pour l'analyse de sentiment, on a employés 2 méthode, la première méthode est proposé par NLTK avec la fonction SentimentIntensityAnalyzer qui nous donne des informations sur les sentiment dans un texte, mais pour appuyer notre analyse on a employée un autre model connnu et souvent utilisé pour l'analyse de sentiment, Roberta, un modèle entrainer avec des donnée provenant de twitter et qui s'avère être très performant pour les données venant d'internet. On a donc fait passer chaque texte dans roberta afin d'en extraire le sentiment, et ensuite on a tout exporter au format CSV pour lasiser place a la visualisation de donnée avec PowerBi 

## La visualisation :

La visualisation est la dernière partie de notre solution, on utilise PowerBi pour crée des dashboard interractif et montrée comment on peut interprété nos données.

