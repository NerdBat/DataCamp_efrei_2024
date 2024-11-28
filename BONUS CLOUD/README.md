Version CLOUD AVEC AWS (avoir un compte AWS)

Le fichier etl_cloud.py se connecte au service Athena d'AWS pour envoyer les fichier CSV dans la base S3. 
Ensuite on se connecte à Redshift pour créer nos tables et insérer les données du fichier CSV de S3.

Si on souhaite automatisé cette taches on peut copié le script dans un job RedShift pour l'automatiser si on veut le lancer tout les x temps.

Par la suite on pourra connecter le dashboard PowerBI directement a la base Redshift afin de récupérer nos données mis à jour en temps réelles
sans devoir le faire manuellement avec le CSV.