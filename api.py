# ── Imports ──────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# ── Charger le modèle et le scaler ───────────────────
with open('model.pkl', 'rb') as f:
    modele = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ── Charger les données ───────────────────────────────
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
df = ratings.merge(movies, on='movieId')

# ── Créer l'application ──────────────────────────────
app = FastAPI()

# ── Format des données reçues ─────────────────────────
class DonneesFilm(BaseModel):
    userId: int
    movieId: int

class DonneesUser(BaseModel):
    userId: int


class NouvelUtilisateur(BaseModel):
    notes: dict

class NouvelUtilisateurRecommander(BaseModel):
    userId: int
# ── Route accueil ─────────────────────────────────────
@app.get("/")
def accueil():
    return {"message": "API de recommandation de films — KNN"}

# ── Route 1 : Prédire une note ────────────────────────
@app.post("/predire")
def predire(donnees: DonneesFilm):
    # Vérifier si le user a déjà noté ce film
    deja_note = df[
        (df['userId'] == donnees.userId) & 
        (df['movieId'] == donnees.movieId)
    ]
    
    if not deja_note.empty:
        note_reelle = deja_note['rating'].values[0]
        return {
            "userId": donnees.userId,
            "movieId": donnees.movieId,
            "message": "Ce film a déjà été noté par cet utilisateur",
            "note_reelle": note_reelle
        }
    
    # Prédire la note
    X = np.array([[donnees.userId, donnees.movieId]])
    X = scaler.transform(X)
    note = modele.predict(X)[0]
    
    return {
        "userId": donnees.userId,
        "movieId": donnees.movieId,
        "note_predite": round(float(note), 2)
    }

# ── Route 2 : Recommander des films ──────────────────
@app.post("/recommander")
def recommander(donnees: DonneesUser):
    # Films déjà vus par l'utilisateur
    films_vus = df[df['userId'] == donnees.userId]['movieId'].tolist()
    
    # Films non vus
    tous_les_films = df['movieId'].unique()
    films_non_vus = [f for f in tous_les_films if f not in films_vus]
    
    # Prédire la note pour chaque film non vu (limité à 500)
    predictions = []
    for film_id in films_non_vus[:500]:
        X = np.array([[donnees.userId, film_id]])
        X = scaler.transform(X)
        note = modele.predict(X)[0]
        predictions.append((film_id, round(float(note), 2)))
    
    # Top 5 des recommandations
    top5 = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
    
    # Ajouter les titres
    resultats = []
    for film_id, note in top5:
        titre = df[df['movieId'] == film_id]['title'].values[0]
        resultats.append({
            "movieId": int(film_id),
            "film": titre,
            "note_predite": note
        })
    
    return {
        "userId": donnees.userId,
        "recommandations": resultats
    }
# ── Route 3 : Ajouter un nouvel utilisateur ───────────
@app.post("/ajouter_utilisateur")
def ajouter_utilisateur(donnees: NouvelUtilisateur):
    global df, modele, scaler
    
    # Créer un nouvel userId unique
    nouvel_id = int(df['userId'].max()) + 1
    
    # Créer les nouvelles lignes
    nouvelles_notes = []
    for movie_id, note in donnees.notes.items():
        nouvelles_notes.append({
            'userId': nouvel_id,
            'movieId': int(movie_id),
            'rating': float(note)
        })
    
    nouveau_df = pd.DataFrame(nouvelles_notes)
    
    # Vérifier que les films existent
    films_valides = df['movieId'].unique()
    films_invalides = [m for m in donnees.notes.keys() 
                      if int(m) not in films_valides]
    if films_invalides:
        return {"erreur": f"Films introuvables : {films_invalides}"}
    
    # Ajouter au dataset
    df = pd.concat([df, nouveau_df.merge(
        movies[['movieId', 'title']], on='movieId', how='left'
    )], ignore_index=True)
    
    # Réentraîner le modèle
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    X = df[['userId', 'movieId']].values
    y = df['rating'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    scaler.transform(X_test)
    
    modele = KNeighborsRegressor(n_neighbors=20)
    modele.fit(X_train, y_train)
    
    return {
        "message": f"Utilisateur {nouvel_id} ajouté avec succès !",
        "userId": nouvel_id,
        "films_notés": len(donnees.notes)
    }

# ── Route 4 : Recommander au nouvel utilisateur ───────
@app.post("/recommander_nouveau")
def recommander_nouveau(donnees: NouvelUtilisateurRecommander):
    # Films déjà vus
    films_vus = df[df['userId'] == donnees.userId]['movieId'].tolist()
    
    if not films_vus:
        return {"erreur": f"Utilisateur {donnees.userId} introuvable"}
    
    # Films non vus
    tous_les_films = df['movieId'].unique()
    films_non_vus = [f for f in tous_les_films if f not in films_vus]
    
    # Prédire
    predictions = []
    for film_id in films_non_vus[:500]:
        X = np.array([[donnees.userId, film_id]])
        X = scaler.transform(X)
        note = modele.predict(X)[0]
        predictions.append((film_id, round(float(note), 2)))
    
    # Top 5
    top5 = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
    
    resultats = []
    for film_id, note in top5:
        titre = df[df['movieId'] == film_id]['title'].values[0]
        resultats.append({
            "movieId": int(film_id),
            "film": titre,
            "note_predite": note
        })
    
    return {
        "userId": donnees.userId,
        "recommandations": resultats
    }