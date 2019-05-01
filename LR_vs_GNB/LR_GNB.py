import spotipy
import pandas as pd
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split 
from logistic_regression import LogisticRegression 
from gaussian_naive_bayes import GaussianNaiveBayes
import matplotlib.pyplot as plt

def get_playlist_tracks(uri, features_of_interest):
    """Retrieves all the songs from a Spotify playlist and its selected features.

    Parameters
    ----------
    uri: string
        Spotify URI code.
    features_of_interest: list
        List of song features to be retrieved. 

    Returns
    -------
    Dataframe
        Table with playlist_id, track_id and each song feature as columns.  
    """
    playlist = sp.user_playlist_tracks(uri.split(':')[2], uri.split(':')[4])
    tracks = playlist['items']
    list_of_ids = []
    list_of_names = []
    features_per_track = {}
    while playlist['next']:
        playlist = sp.next(playlist)
        tracks.extend(playlist['items'])
    for track in tracks:
        list_of_ids.append((track['track']['id']))
        list_of_names.append((track['track']['name']))
    for n, (track_id, track_name) in enumerate(zip(list_of_ids, list_of_names)): 
        features = dict((key,value) for key, value in sp.audio_features(track_id)[0].items() if key in features_of_interest)
        features['track_id'] = track_id
        features['playlist_id'] = uri.split(':')[4]
        features_per_track[n] = features
    return pd.DataFrame.from_dict(features_per_track, orient='index')

try:
    client_credentials_manager = SpotifyClientCredentials(client_id = 'bddfdc9233b5493899809dcc42ca5cc3', client_secret = 'd97a1e581b5f4b4b9da348d6a0529e02')
    sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
    audio_features = ['danceability', 'energy','speechiness', 'acousticness','instrumentalness', 'valence']
    uri_rap = 'spotify:user:spotifycharts:playlist:4NvVpXZLIZ4z5yzTIqgLve'
    uri_jazz = 'spotify:user:spotifycharts:playlist:1Rj92hyXm3WjpOJI8XgYtF'
    rap = get_playlist_tracks(uri_rap, audio_features) ##retrieves 'Best of Rap'
    jazz = get_playlist_tracks(uri_jazz, audio_features) ##retrieves 'Cafe Jazz'
    frames = [rap, jazz]
    result = pd.concat(frames).reset_index().drop('index', 1) ##concatenate the Dataframes
    result.to_csv('dataset.csv', index=False)
except:
    print("No credentials!")

data = pd.read_csv('dataset.csv')
data_trans = data.copy()
data_trans.loc[data['playlist_id'] == '4NvVpXZLIZ4z5yzTIqgLve', 'playlist_id'] = 0 ##rap
data_trans.loc[data['playlist_id'] == '1Rj92hyXm3WjpOJI8XgYtF', 'playlist_id'] = 1 ##jazz
data_trans.loc[:, 'danceability':'valence'].describe()

audio_features = ['danceability', 'energy', 'speechiness', 'acousticness','instrumentalness', 'valence']
all_playlists = data_trans[audio_features].describe().T

rap = data_trans.loc[data_trans['playlist_id'] == 0][audio_features].describe().T
rap.rename(columns={'mean':'mean_rap'}, inplace=True)

jazz = data_trans.loc[data_trans['playlist_id'] == 1][audio_features].describe().T
jazz.rename(columns={'mean':'mean_jazz'}, inplace=True)

df1 = rap['mean_rap']
df2 = jazz['mean_jazz']
df3 = all_playlists['mean']

r = pd.concat([df1, df2, df3], axis=1)
r.plot(kind='bar', figsize=(8,5), title='Audio feature average value per playlist', colormap='viridis', rot=20);

features = data_trans.loc[:, 'danceability':'valence'].values
targets = data_trans.loc[:, 'playlist_id'].values
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.30, random_state=100)

lr = LogisticRegression(iterations=15000, learning_rate=0.10)
pred_y = lr.fit(x_train, y_train).predict(x_test)
accuracy_score(pred_y, y_test)

confusion_matrix(y_test, pred_y)

gnb = GaussianNaiveBayes()
pred_y = gnb.fit(x_train, y_train).predict(x_test)
accuracy_score(y_test, pred_y)

confusion_matrix(y_test, pred_y)