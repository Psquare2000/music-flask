from flask import Flask, request, jsonify
import os

app = Flask(__name__)

import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

url='https://drive.google.com/file/d/1AL44qt1ZCyzmG-1d3J9dqco3-8oOrYYB/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2]


data_songs = pd.read_csv(url)

def find_song(name, year):
    song_data = defaultdict()
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="d5bbe2fe821f412f97e32c51c370f8ce",
                                                        client_secret="9c99ca82d8b04712937a9563a7004f58"))
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=10):
    
    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                ('kmeans', KMeans(n_clusters=20, 
                                verbose=0))
                                ], verbose=0)
    

    X = data_songs.select_dtypes(np.number)
    number_cols = list(X.columns)
    song_cluster_pipeline.fit(X)
    song_cluster_labels = song_cluster_pipeline.predict(X)
    
    data_songs['cluster_label'] = song_cluster_labels
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Welcome to our medium-greeting-api!</h1>"

@app.route('/predict/', methods=['GET'])
def respond():
    name1=request.args.get("name1",None)
    year1=request.args.get("year1",None)
    name2=request.args.get("name2",None)
    year2=request.args.get("year2",None)
    name3=request.args.get("name3",None)
    year3=request.args.get("year3",None)
    name4=request.args.get("name4",None)
    year4=request.args.get("year4",None)
    name5=request.args.get("name5",None)
    year5=request.args.get("year5",None)

    response = {}

    # Check if the user sent a name at all
    if not name1:
        response["ERROR"] = "No name found. Please send a name."
    # # Check if the user entered a number
    # elif str(name).isdigit():
    #     response["ERROR"] = "The name can't be numeric. Please send a string."
    else:

        response["MESSAGE"] = recommend_songs([{'name': name1, 'year':int(year1)},
                {'name': name2, 'year': int(year2)},
                {'name': name3, 'year': int(year3)},
                {'name': name4, 'year': int(year4)},
                {'name': name5, 'year': int(year5)}],  data_songs)


#     Return the response in json format
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
