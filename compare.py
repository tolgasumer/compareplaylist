import sys
import spotipy
import spotipy.util as util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#calculate runtime
import time
start = time.time()

scope = 'user-library-read'
token = util.prompt_for_user_token("11100316938",scope,client_id='bddfdc9233b5493899809dcc42ca5cc3',client_secret='d97a1e581b5f4b4b9da348d6a0529e02',redirect_uri='http://localhost:5000/callback')
spotify = spotipy.Spotify(auth = token)

def getPlaylistFeatures(playlistURI, playlistNum):
    
    userPlaylistResponse = spotify.user_playlist("11100316938", playlistURI)
    playlistName = userPlaylistResponse["name"]
    
    #tracks are returned in this JSON response
    tracksResponse = spotify.user_playlist_tracks("11100316938", playlistURI)
    tracks = tracksResponse["items"]
    
    #instantiate an empty array and place the Tracks URI in this array
    tracksArray = [] 
    for track in tracks:
        URI = track["track"]["id"]
        tracksArray.append(URI)
    
    dfArray = [] #empty array the DataFrame will intake
    dfColumns = ["trackURI", "danceability", "energy", "acousticness", "tempo", "valence", "loudness", "speechiness", "instrumentalness", "playlistNum"] #columns that make DataFrame
        
    #code to iterate through each track URI and get its respective features
    for trackURI in tracksArray[:10]:
        trackFeatures = spotify.audio_features(trackURI)
        
        danceability = trackFeatures[0]["danceability"]
        energy = trackFeatures[0]["energy"]
        acousticness = trackFeatures[0]["acousticness"]
        tempo = trackFeatures[0]["tempo"]
        valence = trackFeatures[0]["valence"]
        loudness = trackFeatures[0]["loudness"]
        speechiness = trackFeatures[0]["speechiness"]
        instrumentalness = trackFeatures[0]["instrumentalness"]
        
        tempArray = [trackURI, danceability, energy, acousticness, tempo, valence, loudness, speechiness, instrumentalness, playlistNum]
        dfArray.append(tempArray)
        
    playlistDF = pd.DataFrame(dfArray, columns = dfColumns)
    return (playlistDF, playlistName)

def mostDissimilarFeatures(playlist1DF, playlist2DF):
#     playlist1 = getPlaylistFeatures(playlist1, 1)
#     playlist2 = getPlaylistFeatures(playlist2, 2)
    
    playlist1Mean = pd.DataFrame(playlist1DF[["danceability", "energy", "acousticness", "valence", "instrumentalness", "speechiness"]].mean(), columns = ["playlist1"])
    playlist2Mean = pd.DataFrame(playlist2DF[["danceability", "energy", "acousticness", "valence", "instrumentalness", "speechiness"]].mean(), columns = ["playlist2"])

    mergedMeans = playlist1Mean.join(playlist2Mean)
    
    differenced = mergedMeans.assign(absoluteDifference = lambda x: np.absolute(mergedMeans["playlist1"] - mergedMeans["playlist2"]))
    differenced = differenced.sort_values("absoluteDifference", ascending = False)

    print(differenced)

    topDissimilar = differenced.index
    
    return topDissimilar

def plotPlaylists(playlist1, playlist2, feature1, feature2):
#     feature1 = "acousticness"
#     feature2 = "instrumentalness"

    playlist1DF = playlist1[0]
    playlist2DF = playlist2[0]
    
    playlist1Name = playlist1[1]
    playlist2Name = playlist2[1]

    plt.scatter(playlist1DF[feature1], playlist1DF[feature2], c = "r")
    plt.scatter(playlist2DF[feature1], playlist2DF[feature2], c = "g")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend(labels = [playlist1Name, playlist2Name])
    plt.show

def KNNPredict(kNeighbors, playlist1, playlist2, feature1, feature2, songFeatures):

    merge = [playlist1[0],playlist2[0]] 
    dfMerge = pd.concat(merge)

    features = dfMerge[[feature1,feature2]]
    classification = dfMerge['playlistNum']
    
    model = KNeighborsClassifier(n_neighbors=kNeighbors)
    model.fit(features, classification)
    
#     return(model)
    return(model.predict([songFeatures]))


def getSongFeature(songURI):
    features = spotify.audio_features(songURI)
    features = features[0]
    
    print(features)
    
    danceability = features["danceability"]
    energy = features["energy"]
    acousticness = features["acousticness"]
    tempo = features["tempo"]
    valence = features["valence"]
    loudness = features["loudness"]
    speechiness = features["speechiness"]
    instrumentalness = features["instrumentalness"]
    
    dfArray = [danceability, energy, acousticness, tempo, valence, loudness, speechiness, instrumentalness] #empty array the DataFrame will intake
    dfColumns = ["danceability", "energy", "acousticness", "tempo", "valence", "loudness", "speechiness", "instrumentalness"] #columns that make DataFrame
    
    trackFeatures = pd.DataFrame(dfArray, index = dfColumns)
    trackFeatures = trackFeatures.transpose()
    return(trackFeatures)

def getSongName(songURI):
    trackName = (spotify.track(songURI)["name"])
    return(trackName)

playlist1Input = '37i9dQZF1DX1rVvRgjX59F' #90s rock
playlist2Input = '37i9dQZF1DXcZDD7cfEKhW' #dance pop

playlist1 = getPlaylistFeatures(playlist1Input, 1)
playlist2 = getPlaylistFeatures(playlist2Input, 2)







#the variables above contain both a DataFrame and the name. We access them by parsing through the returned tuple
playlist1DF = playlist1[0]
playlist1Name = playlist1[1]

playlist2DF = playlist2[0]
playlist2Name = playlist2[1]



print(playlist1Name)
print(playlist1DF.head())

print(playlist2Name)
print(playlist2DF.head())


#DataFrames for each playlist into the function
mostDissimilar = mostDissimilarFeatures(playlist1DF, playlist2DF)

#top 2 dissimilar features
top2Dissimilar = mostDissimilar.values.tolist() #conver to list to access

feature1 = top2Dissimilar[0]
feature2 = top2Dissimilar[1]

print("top2Dissimilar[0]:"+feature1)
print("top2Dissimilar[1]:"+feature2)

plotPlaylists(playlist1, playlist2, feature1, feature2)

from sklearn.neighbors import KNeighborsClassifier
#song = '29R1IMTTbDDA3VNlk6UEW5' #dance
#song = '5CQ30WqJwcep0pYcV4AMNc' #stairway
song = '3e9L9HiHKcfYLAga28Vmcf'

trackFeatures = getSongFeature(song)
trackName = getSongName(song)
trackFeature1 = trackFeatures.loc[0, feature1]
trackFeature2 = trackFeatures.loc[0, feature2]
#values of top two dissimilar features
trackFeatureValues = [trackFeature1, trackFeature2] 

print("trackFeatureValues:")
print(trackFeatureValues)



myModel = KNNPredict(25, playlist1, playlist2, feature1, feature2, trackFeatureValues)
print(myModel)
if(myModel[0] == 1 ):
    print(trackName + "------->" + playlist1Name)
else:
    print(trackName + "------->" + playlist2Name)

print('\n------Runtime:', time.time()-start)