# **Système de Recommandation d'Articles Basé sur les Embeddings Visuels et Textuels**

Ce projet implémente un système de recommandation d'articles en ligne utilisant des embeddings visuels et textuels. Le système permet de rechercher des articles similaires en fonction d'une image, d'une description textuelle ou d'une combinaison des deux. Il est développé en Python en utilisant des modèles de deep learning comme `ResNet50` pour l'extraction d'embeddings visuels et `Sentence-BERT` pour les embeddings textuels.

## **Fonctionnalités**

1. **Recherche par image** : Recherche des articles similaires à partir d'une image téléchargée par l'utilisateur.
2. **Recherche par texte** : Recherche des articles similaires en utilisant une description textuelle.
3. **Recherche combinée** : Combine les résultats des recherches visuelles et textuelles pour recommander des articles pertinents.

## **Prérequis**

- Python 3.x
- Libraries Python :
  - `streamlit` : pour l'interface utilisateur.
  - `tensorflow` : pour l'utilisation de modèles pré-entraînés comme ResNet50.
  - `sentence-transformers` : pour l'extraction d'embeddings textuels.
  - `pandas`, `numpy` : pour la gestion des données et calcul des similarités.
  - `scikit-learn` : pour calculer la similarité cosinus entre les embeddings.

## **Explications supplémentaires**
1. **Modèle ResNet50** : Ce modèle de CNN pré-entrainé est utilisé pour extraire des caractéristiques des images.
2. **Sentence-BERT** : Un modèle pré-entraîné pour générer des embeddings de phrases, qui sont utilisés pour la recherche par texte.
3. **Cosine Similarity** : Les similarités entre les embeddings d'images et de texte sont calculées à l'aide de la fonction de similarité cosinus pour trouver les articles les plus similaires.

## **Lancer l'application Streamlit**

Avant de lancer l'application, assurez-vous que les modèles et les données nécessaires sont disponibles dans le répertoire du projet. Les modèles pré-entraînés, comme **ResNet50** pour les embeddings visuels et **Sentence-BERT** pour les embeddings textuels, seront téléchargés automatiquement lorsque vous exécutez le code, si vous ne les avez pas encore.

### **Étapes pour démarrer l'application**

1. Assurez-vous d'avoir installé toutes les dépendances
2. Exécutez l'application Streamlit :
`streamlit run app.py`

3. Une fois l'application lancée, Vous verrez l'interface utilisateur de l'application ouverte sur votre navigateur qui vous permet de faire des recherches par image, texte ou une combinaison des deux.
