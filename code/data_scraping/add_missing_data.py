import csv
import pandas as pd

from datetime import datetime
from .const import cids
from .script_to_download_table import get_chart_dates, get_top_chart


def fill_out_missing_data():
    url = "https://top40-charts.com/chart.php?cid={}&date={}"
    url_date = "https://top40-charts.com/chart.php?cid={}"
    file_name = "top_charts_missing_data2.csv"
    history_file_name = "C:\\Users\\niew\\OneDrive - Netcompany\\Documents\\private\\masters\\data scraping\\code\\data_scraping\\all_data.csv"
    df = pd.read_csv(history_file_name,encoding='latin-1')

    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Country", "Position", "Date", "Song title", "Song author"])
        for cid in cids.keys():
            try:
                dates = get_chart_dates(url_date.format(cid))
                dates.reverse()
                for dt in dates[:1]:
                    try:
                        chart_date = datetime.strptime(dt, "%d-%m-%Y")
                        # print(chart_date)
                        res = df[(df['Country'] == cids[cid]) & (df['Date'] == chart_date)]
                        if len(res) > 0:
                            continue
                        print(cid, chart_date)
                        country_url = url.format(cid, chart_date.strftime("%Y-%m-%d"))
                        data = get_top_chart(country_url,chart_date.strftime("%Y-%m-%d"))
                        if data is None:
                            continue
                        writer.writerows(data)
                    except:
                        print('error for :' + country_url)
                    break
            except:
                print('error on cid: ' + cid)

fill_out_missing_data()
        