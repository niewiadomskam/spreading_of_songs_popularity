import spotipy
from spotipy.oauth2 import SpotifyOAuth
from code.data_scraping.const import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI

scope = "user-read-recently-played"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID, client_secret= SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=scope))

result = sp.search(q="%20track:Let%20it%20be%20artist:Michael%20Jackson", type="track")
# print(result)
for track in result["tracks"]["items"]:
    print(track["album"])
    res = sp.track(track["id"])
    break
    # print(res["album"])

result = sp.search(q="Ed Sheeran", type="artist")
for artist in result["artists"]["items"]:
    print(artist["id"])
    break