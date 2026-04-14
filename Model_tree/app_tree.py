from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class GenresBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        self.mlb.fit(X.iloc[:, 0])
        return self

    def transform(self, X):
        return self.mlb.transform(X.iloc[:, 0])

app = Flask(__name__)

with open("reco_model.pkl", "rb") as f:
    pipe = pickle.load(f)

ratings = pd.read_csv("ratings.csv")
movies_raw = pd.read_csv("movies.csv")

df = ratings.merge(movies_raw, on="movieId")

df["genres_list"] = df["genres"].apply(lambda x: x.split("|") if isinstance(x, str) else [])
df["like"] = (df["rating"] >= 4).astype(int)

user_mean = df.groupby("userId")["rating"].mean().rename("user_mean_rating")
df = df.merge(user_mean, on="userId")

movies = df.groupby("movieId", as_index=False).agg(
    title=("title", "first"),
    genres_list=("genres_list", "first"),
    movie_mean_rating=("rating", "mean"),
    movie_rating_count=("rating", "count")
)

def recommend_for_user(user_id, df_ratings, movies, pipe, top_n=10):
    user_ratings = df_ratings[df_ratings["userId"] == user_id]

    if user_ratings.empty:
        return None

    seen = set(user_ratings["movieId"])

    candidates = movies[~movies["movieId"].isin(seen)].copy()
    if candidates.empty:
        return candidates

    user_mean = df_ratings.loc[df_ratings["userId"] == user_id, "user_mean_rating"].iloc[0]

    X_cand = candidates.copy()
    X_cand["user_mean_rating"] = user_mean

    feature_cols = ["genres_list", "user_mean_rating", "movie_mean_rating", "movie_rating_count"]
    X_cand = X_cand[feature_cols]

    proba = pipe.predict_proba(X_cand)[:, 1]
    candidates["like_proba"] = proba

    recs = candidates.sort_values(
        ["like_proba", "movie_mean_rating"],
        ascending=False
    ).head(top_n)

    return recs[["movieId", "title", "like_proba", "movie_mean_rating", "movie_rating_count"]]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_id = data.get("userId")

    if user_id is None:
        return jsonify({"error": "userId est requis"}), 400

    try:
        user_id = int(user_id)
    except:
        return jsonify({"error": "userId doit être un entier"}), 400

    recs = recommend_for_user(user_id, df, movies, pipe, top_n=10)

    if recs is None:
        return jsonify({"error": "Utilisateur introuvable"}), 404

    if recs.empty:
        return jsonify({"userId": user_id, "recommendations": []})

    return jsonify({
        "userId": user_id,
        "recommendations": recs.to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run(debug=True)