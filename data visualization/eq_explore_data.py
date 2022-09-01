import json

import pandas as pd
import plotly.express as px

filename = 'data/eq_data_1_day_m1.json'
readablefile = 'data/readable_eq_data.json'
with open(filename) as f:
    all_eq_data = json.load(f)

all_eq_dicts = all_eq_data['features']
# print(len(all_eq_dicts))
# 获取每次地震的震级、位置信息
mags, titles, lons, lats = [], [], [], []
for eq_dict in all_eq_dicts:
    mag = eq_dict['properties']['mag']
    title = eq_dict['properties']['title']
    # 分别获取地震位置的经度和纬度
    lon = eq_dict['geometry']['coordinates'][0]
    lat = eq_dict['geometry']['coordinates'][1]
    mags.append(mag)
    titles.append(title)
    lons.append(lon)
    lats.append(lat)
# print(mags[:10])
data = pd.DataFrame(
    data=zip(lons, lats, titles, mags), columns=['经度', '纬度', '位置', '震级']
)
fig = px.scatter(
    data,
    x='经度',
    y='纬度',
    range_x=[-200, 200],
    range_y=[-90, 90],
    width=800,
    height=800,
    title='全球地震散点图',
    # size='震级',
    # size_max=10,
    color='震级',
    color_continuous_scale='sunset',
    hover_name='位置'
)
fig.write_html('global_earthquakes.html')
fig.show()
