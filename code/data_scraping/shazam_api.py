from shazamio import Shazam, Serialize
import asyncio


async def get_genre(songname, artistname):
  try:
    shazam = Shazam()
    tracks = await shazam.search_track(query=songname, limit=5)
    if "tracks" not in tracks.keys() or len(tracks["tracks"]["hits"]) == 0 :
      return None
    for track in tracks["tracks"]["hits"]:
      if(track["heading"]["subtitle"] == artistname):
          track_id = track["key"]
          about_track = await shazam.track_about(track_id=track_id)
          # print(about_track)
          if "genres" in about_track.keys():
              return about_track["genres"]["primary"] 
          else:
            return None
  except:
     return None