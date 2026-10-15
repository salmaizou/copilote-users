# Analyse des traces Copilote

Ce projet analyse des traces de sessions Copilote. Il permet de préparer un classeur Excel, de produire des statistiques descriptives et d'entraîner un modèle qui prédit l'utilisateur associé à une session.

> Les fichiers de données et les résultats générés ne sont pas inclus dans le dépôt. Il faut ajouter les fichiers CSV localement avant de lancer les scripts.

## Contenu du Projet

- `main.py` : rassemble les CSV dans un classeur Excel pour faciliter leur consultation.
- `statistiques.py` : calcule des statistiques sur les navigateurs, les utilisateurs, les actions et les durées estimées. Génère des tableaux CSV et un classeur Excel.
- `modele.py` : évalue un classifieur Bayes naïf sur une séparation stratifiée du jeu d'entraînement, puis entraîne le modèle sur toutes les données d'entraînement et prédit les utilisateurs des sessions de test.
- `requirements.txt` : dépendance Python nécessaire à la création des fichiers Excel (`openpyxl`).

 du dépôt avec `.gitignore`. Ne retirez pas ces exclusions pour publier des données privées ou des fichiers de compétition.
