🎬 IA Movie Recommender – De zéro au cloud
Ce projet est un système de recommandation de films basé sur l’IA, pensé comme un mini pipeline complet : des données brutes jusqu’à une API déployée en ligne.

🎯 Objectif du projet
Partir d’un jeu de données de films (titres, genres, notes, utilisateurs).

Construire un modèle d’IA capable de recommander des films pertinents à un utilisateur.

Mettre ce modèle à disposition via une API simple, accessible depuis n’importe quelle appli ou script.

🧹 Étape 1 – Préparation des données
Nettoyage des données brutes (CSV).

Gestion des valeurs manquantes et des doublons.

Encodage et mise en forme des variables utiles (films, utilisateurs, genres, notes).

Constitution d’un jeu de données propre, prêt pour l’entraînement du modèle.

🧠 Étape 2 – Modèle de recommandation
Création et entraînement d’un modèle de recommandation (type filtrage collaboratif / basé contenu, selon les tests).

Évaluation rapide de la qualité des recommandations sur un jeu de validation.

Sauvegarde du modèle pour pouvoir le réutiliser facilement dans l’API.

🌐 Étape 3 – API d’inférence
Création d’une mini API avec FastAPI ou Flask.

L’API reçoit une requête (par exemple un user_id et un nombre de recommandations souhaité) et renvoie une liste de films recommandés.

Pensée pour être simple à appeler depuis un script, une appli web ou un outil de tests.

☁️ Étape 4 – Déploiement dans le cloud
Déploiement de l’API sur un service gratuit : Azure, Replit ou HuggingFace Spaces (selon la solution choisie).

Mise à disposition d’une URL publique permettant de tester le système depuis n’importe où.

Objectif : rendre le modèle utilisable comme un vrai service en ligne, pas seulement en local.

🧪 Tests et validation
Envoi de requêtes vers l’API (ex. via un client HTTP, Postman, curl ou script Python).

Vérification que les données envoyées sont correctement traitées et que l’API renvoie des recommandations cohérentes.

Ajustements possibles du modèle ou du prétraitement en fonction des résultats.

🔮 Pistes d’amélioration
Améliorer la qualité des recommandations (meilleurs algorithmes, features plus riches).

Ajouter des filtres (genre, langue, durée, popularité).

Intégrer une interface utilisateur simple pour tester le système sans écrire de code.
