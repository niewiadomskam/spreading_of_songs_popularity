import requests
import time
import asyncio
import csv
import pandas as pd

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from .const import cids
from .shazam_api import get_genre
from .get_artist_metadata import get_artist_country


def get_top_chart(url, chart_date, df=None):
    # print(url)
    try:
        count = 0
        while count < 10:
            response_body = requests.get(url)
            if response_body.status_code == 200: 
                break
            time.sleep(1)  # import time
        if response_body.status_code != 200:
            print('cannot load page: '+ url)
            raise Exception('cannot load page')
        soup = BeautifulSoup(response_body.text, "lxml")
        country = soup.find("font").text.split(' ')
        country = get_country_name(country)
        if df is not None:
            res = df[(df['Country'] == country) & (df['Date'] == chart_date)]
            if len(res) > 0:
                return None
        chart_table = soup.find('table', attrs={"cellpadding": "2"})
        songs_rows = chart_table.find_all("tr", class_="latc_song")

        chart_data = []
        for row in songs_rows:
            position = row.find("td", class_="text-nowrap text-center")
            song_data = row.find("table")
            
            song_title = song_data.find("div", attrs={"style":"margin-bottom: 4px;"})
            song_author = song_data.find("a", attrs={"style":"text-decoration: none; "})
            # print(song_title.text, song_author.text)
            # loop = asyncio.get_event_loop()
            # genre = loop.run_until_complete(get_genre(song_title.text, song_author.text))
            # artist_country = get_artist_country(song_author.text)
            # print(genre)
            chart_data.append([country, position.text, chart_date, song_title.text, song_author.text])
        return chart_data
    except:
        print('cannot load')

def get_start_date(url):
    response_body = requests.get(url)
    soup = BeautifulSoup(response_body.text, "lxml")
    select_date = soup.find("select", attrs={"name": "date"})
    date_selected = select_date.find("option", selected=True)
    return date_selected.text

def get_chart_dates(url):
    response_body = requests.get(url)
    soup = BeautifulSoup(response_body.text, "lxml")
    select_date = soup.find("select", attrs={"name": "date"})
    dates_option = select_date.find_all("option")
    result = []
    for dt in dates_option:
        result.append(dt.text)
    return result

def get_country_name(country):
    if country[1]=='Top' :
        return country[0]
    else:
        idx = country.index('Top')
        return ' '.join(country[:idx])

def scrape_data():
    url = "https://top40-charts.com/chart.php?cid={}&date={}"
    url_date = "https://top40-charts.com/chart.php?cid={}"
    file_name = "top_charts_only_countries5.csv"
    # song_data = []
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Country", "Position", "Date", "Song title", "Song author"])
        for cid in cids:
            try:
                # print('here1')
                # start_date = get_start_date(url_date.format(cid))
                # start_date= datetime.strptime(start_date, "%d-%m-%Y")
                dates = get_chart_dates(url_date.format(cid))
                # print('here2')
                dates.reverse()
                # print(dates)
                for dt in dates:
                    try:
                        # print('here3')
                        chart_date = datetime.strptime(dt, "%d-%m-%Y")
                        # print('here4')
                        print(cid, chart_date)
                        country_url = url.format(cid, chart_date.strftime("%Y-%m-%d"))
                        # print('here5', country_url)
                        data = get_top_chart(country_url,chart_date.strftime("%Y-%m-%d"))
                        writer.writerows(data)
                    except:
                        print('error for :' + country_url)
            except:
                print('error on cid: ' + cid)
        
                # song_data.extend(data)
    # df = pd.DataFrame(columns = ["Country", "Position", "Date", "Song title", "Song author", "Country", "Genre"], data = song_data)
    # df.to_csv(file_name)



                # song_data.extend(data)
    # df = pd.DataFrame(columns = ["Country", "Position", "Date", "Song title", "Song author", "Country", "Genre"], data = song_data)
    # df.to_csv(file_name)


# scrape_data()