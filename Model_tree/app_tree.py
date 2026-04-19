from flask import Flask, jsonify, render_template
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

ratings = pd.read_csv("ratings.csv")
movies_df = pd.read_csv("movies.csv")

df = ratings.merge(movies_df, on="movieId", how="left")

df["genres_list"] = df["genres"].str.split("|")
df["like"] = (df["rating"] >= 3).astype(int)

user_stats = df.groupby("userId")["rating"].mean().rename("user_mean_rating").reset_index()
df = df.merge(user_stats, on="userId", how="left")

movie_stats = df.groupby("movieId")["rating"].agg(
    movie_mean_rating="mean",
    movie_rating_count="count"
).reset_index()
df = df.merge(movie_stats, on="movieId", how="left")

movies = df.groupby("movieId", as_index=False).agg(
    title=("title", "first"),
    genres_list=("genres_list", "first"),
    movie_mean_rating=("movie_mean_rating", "first"),
    movie_rating_count=("movie_rating_count", "first")
)

class GenresBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        self.mlb.fit(X.iloc[:, 0])
        return self

    def transform(self, X):
        return self.mlb.transform(X.iloc[:, 0])

feature_cols = ["genres_list", "user_mean_rating", "movie_mean_rating", "movie_rating_count"]
X = df[feature_cols]
y = df["like"]

preprocess = ColumnTransformer(
    transformers=[
        ("genres", GenresBinarizer(), ["genres_list"]),
        ("num", "passthrough", ["user_mean_rating", "movie_mean_rating", "movie_rating_count"])
    ]
)

clf = DecisionTreeClassifier(
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", clf)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(X_train, y_train)
print("Accuracy:", pipe.score(X_test, y_test))


def recommend_for_user(user_id, df_ratings, movies, pipe, top_n=10, min_votes=20):
    user_ratings = df_ratings[df_ratings["userId"] == user_id]
    seen = set(user_ratings["movieId"])

    min_votes = 3
    candidates = movies[
        (~movies["movieId"].isin(seen)) &
        (movies["movie_rating_count"] >= min_votes)
    ].copy()

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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend/<int:user_id>")
def recommend(user_id):
    if user_id not in df["userId"].values:
        return jsonify({"error": "userId introuvable"}), 404

    recs = recommend_for_user(user_id, df, movies, pipe, top_n=10)

    if recs is None or recs.empty:
        return jsonify({
            "userId": user_id,
            "recommendations": []
        })

    return jsonify({
        "userId": user_id,
        "recommendations": recs.to_dict(orient="records")
    })


if __name__ == "__main__":
    app.run(debug=True)