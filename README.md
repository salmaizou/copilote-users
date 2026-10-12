# Analyse des traces Copilote

Ce projet analyse des traces de sessions Copilote. Il permet de préparer un classeur Excel, de produire des statistiques descriptives et d'entraîner un modèle qui prédit l'utilisateur associé à une session.

> Les fichiers de données et les résultats générés ne sont pas inclus dans le dépôt. Il faut ajouter les fichiers CSV localement avant de lancer les scripts.

## Contenu du projet

- `main.py` : rassemble les CSV dans un classeur Excel pour faciliter leur consultation.
- `statistiques.py` : calcule des statistiques sur les navigateurs, les utilisateurs, les actions et les durées estimées. Génère des tableaux CSV et un classeur Excel.
- `modele.py` : évalue un classifieur Bayes naïf sur une séparation stratifiée du jeu d'entraînement, puis entraîne le modèle sur toutes les données d'entraînement et prédit les utilisateurs des sessions de test.
- `requirements.txt` : dépendance Python nécessaire à la création des fichiers Excel (`openpyxl`).

## Installation

Ouvrez PowerShell dans le dossier du projet et créez un environnement virtuel :

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell bloque l'activation de l'environnement, vous pouvez autoriser les scripts pour la session en cours, puis réessayer :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Données attendues

Placez les CSV dans le dossier `data` à la racine du projet :

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

`train.csv` et `test.csv` doivent être des fichiers sans ligne d'en-tête, avec une session par ligne :

- `train.csv` : identifiant utilisateur, navigateur, puis les événements de la session.
- `test.csv` : navigateur, puis les événements de la session.
- `sample_submission.csv` : modèle de soumission fourni avec les données ; `main.py` l'ajoute au classeur Excel.

Les événements peuvent contenir des marqueurs de temps comme `t120`. Le modèle s'en sert pour estimer la durée d'une session.

## Exécuter les scripts

Depuis la racine du projet, avec l'environnement virtuel activé :

```powershell
python main.py
python statistiques.py
python modele.py
```

Chaque script peut aussi être exécuté séparément :

- `main.py` crée `data\donnees_preparees.xlsx`.
- `statistiques.py` crée des tableaux CSV et `data\statistiques\statistiques_descriptives.xlsx`.
- `modele.py` affiche les résultats de validation et crée les fichiers dans `data\modele`.

## Résultats du modèle

Le script `modele.py` génère :

- `data\modele\evaluation.json` : accuracy et F1 macro sur la validation, ainsi que la distribution des prédictions sur le test.
- `data\modele\validation_split.csv` : indique quelles sessions ont été utilisées pour l'apprentissage et la validation.
- `data\modele\submission_baseline.csv` : prédictions pour les sessions de `test.csv`, au format `RowId,prediction`.

La validation utilise une seule séparation stratifiée avec une graine fixe (`42`) et réserve environ 25 % des sessions pour la validation. Les métriques obtenues mesurent le résultat sur cette séparation ; elles ne garantissent pas la performance sur un jeu de test externe. Vérifiez le format demandé par la compétition avant d'utiliser le fichier de prédictions comme soumission.

## Données et fichiers ignorés par Git

Les données, les classeurs générés, les sorties du modèle et l'environnement virtuel sont exclus du dépôt avec `.gitignore`. Ne retirez pas ces exclusions pour publier des données privées ou des fichiers de compétition.
