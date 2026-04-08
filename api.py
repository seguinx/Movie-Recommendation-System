# ── Imports ──────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ── Charger le modèle et la matrice ──────────────────
with open('model.pkl', 'rb') as f:
    knn_matrice = pickle.load(f)

with open('matrice.pkl', 'rb') as f:
    matrice = pickle.load(f)

# ── Charger les données ───────────────────────────────
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
df = ratings.merge(movies, on='movieId')

# ── Créer l'application ──────────────────────────────
app = FastAPI()

# ── Classes ───────────────────────────────────────────
class DonneesFilm(BaseModel):
    userId: int
    movieId: int

class DonneesUser(BaseModel):
    userId: int

class NouvelUtilisateur(BaseModel):
    notes: dict[str, float]

# ── Fonction de recommandation centrale ──────────────
def get_recommandations(userId, n=5):
    user_index = matrice.index.get_loc(userId)
    
    distances, indices = knn_matrice.kneighbors(
        matrice.iloc[user_index].values.reshape(1, -1),
        n_neighbors=20
    )
    
    films_vus = set(matrice.columns[matrice.iloc[user_index] > 0])
    recommandations = {}
    
    for i in range(1, len(indices[0])):
        voisin_index = indices[0][i]
        similarite = 1 - distances[0][i]
        voisin_notes = matrice.iloc[voisin_index]
        
        for movie_id, note in voisin_notes.items():
            if movie_id not in films_vus and note >= 3.5:
                if movie_id not in recommandations:
                    recommandations[movie_id] = 0
                recommandations[movie_id] += note * similarite
    
    top = sorted(recommandations.items(), key=lambda x: x[1], reverse=True)[:n]
    
    resultats = []
    for movie_id, score in top:
        titre = df[df['movieId'] == movie_id]['title'].values[0]
        resultats.append({
            "movieId": int(movie_id),
            "film": titre,
            "score": round(score, 2)
        })
    
    return resultats

# ── Route accueil ─────────────────────────────────────
@app.get("/")
def accueil():
    return {"message": "API de recommandation de films — KNN"}

# ── Route 1 : Recommander à un utilisateur existant ──
@app.post("/recommander")
def recommander(donnees: DonneesUser):
    if donnees.userId not in matrice.index:
        return {"erreur": f"Utilisateur {donnees.userId} introuvable"}
    
    resultats = get_recommandations(donnees.userId)
    return {"userId": donnees.userId, "recommandations": resultats}

# ── Route 2 : Ajouter un nouvel utilisateur ───────────
@app.post("/ajouter_utilisateur")
def ajouter_utilisateur(donnees: NouvelUtilisateur):
    global matrice, knn_matrice, df
    
    # Nouvel userId unique
    nouvel_id = int(matrice.index.max()) + 1
    
    # Créer une nouvelle ligne dans la matrice
    nouvelle_ligne = pd.Series(0.0, index=matrice.columns)
    for movie_id, note in donnees.notes.items():
        mid = int(movie_id)
        if mid in matrice.columns:
            nouvelle_ligne[mid] = float(note)
    
    # Ajouter à la matrice
    nouvelle_ligne.name = nouvel_id
    matrice = pd.concat([matrice, nouvelle_ligne.to_frame().T])
    
    # Ajouter au df pour les titres
    nouvelles_notes = []
    for movie_id, note in donnees.notes.items():
        mid = int(movie_id)
        titre_row = df[df['movieId'] == mid]['title']
        titre = titre_row.values[0] if len(titre_row) > 0 else "Inconnu"
        nouvelles_notes.append({
            'userId': nouvel_id,
            'movieId': mid,
            'rating': float(note),
            'title': titre
        })
    df = pd.concat([df, pd.DataFrame(nouvelles_notes)], ignore_index=True)
    
    # Réentraîner KNN
    knn_matrice = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    knn_matrice.fit(matrice.values)
    
    return {
        "message": f"Utilisateur {nouvel_id} ajouté avec succès !",
        "userId": nouvel_id,
        "films_notés": len(donnees.notes)
    }

# ── Route 3 : Recommander au nouvel utilisateur ───────
@app.post("/recommander_nouveau")
def recommander_nouveau(donnees: DonneesUser):
    if donnees.userId not in matrice.index:
        return {"erreur": f"Utilisateur {donnees.userId} introuvable"}
    
    resultats = get_recommandations(donnees.userId)
    return {"userId": donnees.userId, "recommandations": resultats}