import flask
import pickle
import numpy as np
from typing import List
from sklearn.neighbors import NearestNeighbors
from flask import (
    render_template,
    jsonify
)


class Config(object):
    """Parent configuration class."""
    DEBUG = True
    TESTING = True


class ProductionConfig(Config):
    """Production configuration class"""
    DEBUG = False
    TESTING = False


class DevelopmentConfig(Config):
    """Development configuration class"""
    DEBUG = True
    TESTING = True


app = flask.Flask(__name__)
# choose configuration based on environment variable
app.config.from_object({
    'development': DevelopmentConfig,
    'production': ProductionConfig,
}[app.config['ENV']])

# Load machine learning data
anime_names = pickle.load(open('anime_names.pkl', 'rb'))
user_ids = pickle.load(open('user_ids.pkl', 'rb'))
collab_mat = pickle.load(open('collab_mat.pkl', 'rb'))
model = pickle.load(open('lightfm_model.pkl', 'rb'))

# build mapping of anime names and user ids to index
name_to_index = {}
id_to_index = {}
# map unique shows that are rated by user to an index
for i, show in enumerate(anime_names):
    name_to_index[show] = i
# map user id to index
for i, id in enumerate(user_ids):
    id_to_index[id] = i
# create k nearest neighbor model to query for most similar user
knn_collab = NearestNeighbors(
    n_neighbors=5,
    algorithm='brute',
    metric='euclidean'
).fit(collab_mat)

if app.config['DEBUG']:
    print("DEBUG:")
    print(type(anime_names))
    print(type(user_ids))
    print(type(collab_mat))
    print(type(model))


def _create_new_user_vect(name_dict, rating_dict):
    """
    This function takes a dictionary of the following
    format: { 'Naruto': 8, 'No Game No Life': 7 }
    and create a vector of size 1 x n_shows
    """
    new_vect = np.zeros((1, len(name_dict)))
    for show in rating_dict:
        new_vect[0, name_dict[show]] = rating_dict[show]
    return new_vect


def _recommend_new_user(
    rating_dict,
    data=collab_mat,
    knn_model=knn_collab,
    name_dict=name_to_index,
    name_list=anime_names,
    id_dict=id_to_index
) -> List[str]:
    x = _create_new_user_vect(name_dict, rating_dict)
    most_similar_user_id = knn_model.kneighbors(
        x, n_neighbors=1, return_distance=False)[0][0]
    uid = id_dict[most_similar_user_id]

    _, n_items = data.shape

    scores = model.predict(uid, np.arange(n_items))
    top_anime = name_list[np.argsort(-scores)]

    return top_anime


@app.route('/')
def index() -> str:
    return render_template('index.html')


@app.route('/recommend_new', methods=['GET', 'POST'])
def recommend_new() -> object:
    """
    This function handles the event when a user submits
    their rating for anime
    """
    return jsonify({'result': 'OK'})


if __name__ == "__main__":
    app.run(debug=True)
