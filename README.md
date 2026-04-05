Créer, entraîner et déployer un modèle de Machine Learning dans le
Cloud
Choisissez 1 projet parmi les 4 suivants :
1. Détection de fraude dans les transactions financières
2. Système de recommandation de films
3. Chatbot (assistant virtuel)
4. Reconnaissance d’images
Étape 1 – Comprendre le projet et choisir les données
Ce que vous devez faire :
• Lire sur le projet choisi (contexte, objectif)
• Trouver un jeu de données sur Internet (Kaggle, Hugging Face, etc.)
• Résumer le but du projet en 3-5 lignes
Livrable à remettre :
• Une page Word ou PDF :
o Sujet choisi
o Objectif du projet
o Lien vers le jeu de données utilisé
o Exemple de données (5 lignes ou 5 exemples)
Exemples :
Projet Exemple de données
Fraude Montant, heure, pays, transaction suspecte ou non
Films ID utilisateur, ID film, note
Chatbot Phrase utilisateur, intention (ex : "bonjour" → salutation)
Images Image de chien/chat avec étiquette correspondante
Étape 2 – Préparer les données
Ce que vous devez faire :
• Ouvrir les données avec Python (Pandas)
• Nettoyer (ex : retirer les lignes vides)
• Afficher quelques statistiques et graphiques simples
Livrable à remettre :
• Un notebook Jupyter (.ipynb) contenant :
o Ouverture du fichier
o Affichage de 5 premières lignes
o Graphiques (barres, histogrammes...)
o Résumé des colonnes importantes
Exemples :
Projet Nettoyage
Fraude Normaliser les montants
Films Regrouper les notes des utilisateurs
Chatbot Supprimer la ponctuation inutile
Images Redimensionner toutes les images à 100x100 pixels
Étape 3 – Créer et entraîner un modèle
Ce que vous devez faire :
• Choisir un algorithme simple (ex : arbre de décision, SVM, ou CNN pour images)
• Diviser les données en données d'entraînement/test
• Entraîner le modèle avec scikit-learn ou TensorFlow
Livrable à remettre :
• Un notebook Jupyter avec :
o Entraînement du modèle
o Affichage des résultats (accuracy, confusion matrix, etc.)
o Sauvegarde du modèle entraîné (model.pkl, model.h5)
Exemples :
Projet Modèle
Fraude Arbre de décision ou forêt aléatoire
Films Système de moyenne ou filtrage collaboratif
Chatbot Classifieur simple avec scikit-learn
Images Réseau CNN avec Keras (2-3 couches max)
Étape 4 – Déployer le modèle dans le Cloud
Ce que vous devez faire :
• Créer une mini API (FastAPI ou Flask)
• Déployer cette API dans un service gratuit (Azure, Replit, ou HuggingFace Spaces)
• Tester une requête : envoyer des données → recevoir une réponse
Livrable à remettre :
• Lien vers l’API déployée
• Vidéo ou capture d’écran montrant le test
• Fichier .py de l’API
Exemples :
Projet Exemple de requête
Fraude Envoyer une transaction → Obtenir "Fraude" ou "Non fraude"
Films Envoyer un ID utilisateur → Obtenir 5 films recommandés
Chatbot Envoyer "bonjour" → Obtenir "Salut, comment puis-je t’aider ?"
Images Envoyer une image → Obtenir "Chat" ou "Chien"
Étape 5 – Présenter le projet
Ce que vous devez faire :
• Préparer une présentation PowerPoint de 5 à 7 diapositives
• Expliquer les grandes étapes :
1. Sujet choisi
2. Données
3. Modèle utilisé
4. Résultats
5. Démonstration du déploiement
Livrable à remettre :
• Le fichier PowerPoint
• Les livrables par étape
• Présentation orale (10-15 minutes)
Résumé des livrables par étape
Étape Livrable
1. Choix du projet Document PDF avec le sujet + jeu de données
2. Préparation des données Notebook Jupyter .ipynb
3. Entraînement du modèle Notebook .ipynb + fichier .pkl ou .h5
4. Déploiement Cloud Lien vers API + script .py + test vidéo
5. Présentation PowerPoint + passage à l’oral