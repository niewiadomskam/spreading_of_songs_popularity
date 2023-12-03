import musicbrainzngs as mbz 

mbz.set_useragent('GetArtistsCountryAndGenre.io', '0.1')

def get_artist_country(artist_name):
    artist_list = mbz.search_artists(query=artist_name, limit=5)['artist-list'] 
    if len(artist_list) == 0:
        return None 
    artist = artist_list[0]
    if "country" in artist.keys():
        return artist["country"]
    elif 'area' in artist.keys() and 'name' in artist['area']:
        return artist['area']['name']
    return None

def get_release_date(artist_name, song_title):
    result = mbz.search_recordings(query=song_title, artist=artist_name)
    if 'recording-list' in result:
        recordings = result['recording-list']
        if recordings:
            first_recording = recordings[0]
            if 'release-list' in first_recording:
                releases = first_recording['release-list']
                if releases:
                    release_date = releases[0]
                    if 'date' in release_date:
                        return release_date['date']
    return None