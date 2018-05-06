import flask
import pickle
from flask import (
    render_template,
    jsonify
)


class Config(object):
    """Parent configuration class."""
    DEBUG = True
    TESTING = True


app = flask.Flask(__name__)
app.config.from_object(Config)

# Load machine learning data
anime_names = pickle.load(open('anime_names.pkl', 'rb'))
user_ids = pickle.load(open('user_ids.pkl', 'rb'))
collab_mat = pickle.load(open('collab_mat.pkl', 'rb'))
model = pickle.load(open('lightfm_model.pkl', 'rb'))

if app.config['DEBUG']:
    print("DEBUG:")
    print(type(anime_names))
    print(type(user_ids))
    print(type(collab_mat))
    print(type(model))


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
