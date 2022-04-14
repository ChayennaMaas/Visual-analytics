#!/usr/bin/env python
# coding: utf-8

# # Eindopdracht - Visual analytics

# Gemaakt door: Anna de Geeter, Chayenna Maas

# ## Importeer data
# !pip install dash
# !pip install dash-bootstrap-components


# Standard packages
import json
import requests
import streamlit as st
from streamlit_folium import folium_static

# Libs to deal with tabular data
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import geopandas as gpd

# Plotting packages
import plotly.express as px
import plotly.graph_objects as go

# Lib to create maps
import folium 
from folium import Choropleth, Circle, Marker

# To display stuff in notebook
from IPython.display import display, Markdown


# ## Inspecteer data

pollution=pd.read_csv('south-korean-pollution-data.csv')
pollution.head()

print('Pollution dataframe: ', pollution.shape)

print(pollution.info())

pollution.describe()


# ## 1D inspecties

summary=pd.read_csv('Measurement_summary.csv')

pollution['date']=pd.to_datetime(pollution['date'])


#Maak histogram aan om de type luchtvervuilers te inspecteren
histo= px.histogram(pollution, x=['co','so2', 'no2', 'o3'],
                   barmode='overlay', range_x=[0,75], color_discrete_sequence=['orange', 'lightblue', 'tomato', 'lime'], 
                   opacity=0.6)

#Update de de layout
histo.update_layout(title='<b>Metingen in Zuid-Korea<b>',xaxis_title='Hoeveel gemeten [ppm]', 
                   yaxis_title='Hoevaak gemeten', height=800, width=800)
histo.update_xaxes(nticks=15)


histo.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="down",
            active=0,
            x=1.1,
            y=0.5,
            buttons=list([
                dict(label="All",
                     method="update",
                     args=[{"visible": [True]},
                           {"title": "<b>Alle metingen in Zuid-Korea<b>"}]),
                dict(label="Co",
                     method="update",
                     args=[{"visible": [True, False, False, False]},
                           {"title": "<b>Co2 metingen in Zuid-Korea<b>"}]),
                dict(label="So2",
                     method="update",
                     args=[{"visible": [False,True, False, False]},
                           {"title": "<b>So2 metingen in Zuid-Korea<b>"}]),
                dict(label="No2",
                     method="update",
                     args=[{"visible": [False, False, True, False]},
                           {"title": "<b>No2 metingen in Zuid-Korea<b>"}]),
                dict(label="O3",
                     method="update",
                     args=[{"visible": [False, False, False, True]},
                           {"title": "<b>O3 metingen in Zuid-Korea<b>"}])]))])

#Voeg notaties toe
histo.add_annotation(x=60, y=7000,
            text="<b>Gemiddelde Co: 4.55 [ppm]<b>",
            showarrow=False)
histo.add_annotation(x=60, y=6700,
            text="<b>Gemiddelde So2: 3.55 [ppm]<b>",
            showarrow=False)
histo.add_annotation(x=60, y=6400,
            text="<b>Gemiddelde No2: 14.8 [ppm]<b>",
            showarrow=False)
histo.add_annotation(x=60, y=6100,
            text="<b>Gemiddelde O3: 34.51 [ppm]<b>",
            showarrow=False)

histo.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     step="hour",
                     stepmode="backward"),
            ])
        ),
        rangeslider=dict(
            visible=True
        )
    ))

# histo.show()

####Bar plot aanmaken om de particle matter 2.5 en 10 te visualiseren

pollution['year']=pollution['date'].dt.year
pollution['month']=pollution['date'].dt.month


pm25=summary[summary['PM2.5'] < 110]
pm10=summary[summary['PM10'] < 110]


#Maak boxplot per stof gehalte
bar = go.Figure()

bar.add_trace(go.Box(y=pm25['PM2.5'], name ='PM 2.5 [µm/m3]'))
bar.add_trace(go.Box(y=pm10['PM10'], name='PM 10 [µm/m3]'))

bar.update_layout(barmode='overlay', title='<b>Particulate Matter in Zuid-Korea [µm/m3]<b>', height=805, width=750, boxgroupgap=0.8,boxgap=0.1)

bar.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="down",
            active=0,
            x=1.15,
            y=0.5,
            buttons=list([
                dict(label="Both",
                     method="update",
                     args=[{"visible": [True]},
                           {"title": "<b>Particulate Matter in Zuid-Korea [µm/m3]<b>"}]),
                dict(label="pm2.5",
                     method="update",
                     args=[{"visible": [True, False]},
                           {"title": "<b>Particulate Matter 2.5 in Zuid-Korea [µm/m3]<b>"}]),
                dict(label="pm10",
                     method="update",
                     args=[{"visible": [False,True]},
                           {"title": "<b>Particulate Matter 10 in Zuid-Korea [µm/m3]<b>"}])]))])

# bar.show()


# ## 2D inspecties

####Scatter plot vervuiling per stad in Zuid-korea
scatter = px.scatter(pollution, x="date", y=['co','no2','so2','o3'], color="City", title='<b>Metingen per stad<b>')

scatter.update_layout(height=400, width=850,
    updatemenus=[
        dict(
            active=0,
            x=-0.1,
            y=1,
            buttons=list([
                dict(label="co",
                     method="update",
                     args=[{"visible": [True, False,False,False]},
                           {"title": "<b>Meting CO per stad<b>"}]),
                dict(label="no2",
                     method="update",
                     args=[{"visible": [False, True,False,False]},
                           {"title": "<b>Meting No2 per stad<b>",
                            }]),
                dict(label="so2",
                     method="update",
                     args=[{"visible": [False, False,True,False]},
                           {"title": "<b>Meting So2 per stad<b>",
                            }]),
                dict(label="o3",
                     method="update",
                     args=[{"visible": [False,False, False,True]},
                           {"title": "<b>Meting O3 per stad<b>",
                            }])]))])

scatter.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     step="year",
                     stepmode="backward")])),
        rangeslider=dict(
            visible=True)))

scatter.update_xaxes(title_text='Datum')
scatter.update_yaxes(title_text='Waarde [ppm]')
scatter.update_xaxes(nticks=12)

# scatter.show()


summary['date']=pd.to_datetime(summary['Measurement date'])

df_date = summary['Measurement date'].str.split(" ", n=1, expand=True)


df_date = summary['Measurement date'].str.split(" ", n=1, expand=True)

summary['date'] = df_date[0]
summary['time'] = df_date[1]
summary = summary.drop(['Measurement date'], axis=1)

df_0 = summary.groupby(['date'], as_index=False).agg({'SO2':'mean', 'NO2':'mean', 'O3':'mean', 'CO':'mean', 'PM10':'mean', 'PM2.5':'mean'})
df_0.head()
df_air = df_0.corr()
df_air = summary.groupby(['date'], as_index=False).agg({'SO2':'mean', 'NO2':'mean', 'O3':'mean', 'CO':'mean', 'PM10':'mean', 'PM2.5':'mean'})


fig_line = px.line(df_air, x='date', y=['CO','NO2','SO2','O3'], title="<b>Metingen [ppm] in Seoul<b>",labels={'date': 'Datum', 'variable':'Type:', 'value': 'Waarde [ppm]'})

fig_line.update_layout(height=400, width=850,
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="CO",
                     method="update",
                     args=[{"visible": [True,False,False,False]},
                           {"title": "<b>Gemeten Co [ppm] in Seoul<b>"}]),
                dict(label="NO2",
                     method="update",
                     args=[{"visible": [False,True,False,False]},
                           {"title": "<b>Gemeten No2 [ppm] in Seoul<b>",
                            }]),
                dict(label="SO2",
                     method="update",
                     args=[{"visible": [False,False,True,False]},
                           {"title": "<b>Gemeten So2 [ppm] in Seoul<b>",
                            }]),
                dict(label="O3",
                     method="update",
                     args=[{"visible": [False,False,False,True]},
                           {"title": "<b>Gemeten O3 [ppm] in Seoul<b>",
                            }]),]),
            ),
    ])
fig_line.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     step="year",
                     stepmode="backward"),
            ])
        ),
        rangeslider=dict(
            visible=True
        )
    ))

fig_line.update_xaxes(nticks=12)

  
# fig_line.show()


# ## Geospatiale inspectie

# Lees Air Pollution in Seoul in
stations = pd.read_csv('Measurement_station_info.csv')
measurements = pd.read_csv('Measurement_info.csv')
items = pd.read_csv('Measurement_item_info.csv')

#Laad geojson file in
district_borders=gpd.read_file('https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_topo_simple.json')

district_borders.crs = "EPSG:4326"

# Eenheid aan het item toevoegen
items['Item name (unit)'] = items['Item name'] + ' (' + items['Unit of measurement'].str.lower() + ')'

# Een dict maken van de item codes naar item namen.
items_dict = {row['Item code']: row['Item name (unit)'] for idx, row in items.iterrows()}

# Functie of het goed normaal, slecht of heel slecht is in metingen
def evaluation_generator(good, normal, bad, vbad):
    def measurement_evaluator(value):
        if(pd.isnull(value) or value < 0):
            return np.nan
        elif(value <= good):
            return 'Good'
        elif(value <= normal):
            return 'Normal'
        elif(value <= bad):
            return 'Bad'
        else:
            return 'Very bad'
        
    return measurement_evaluator

#een dictionary dat verontreinigende stoffen toewijst aan functies die het meetniveau evalueren.
evaluators = {
    row['Item name (unit)']: evaluation_generator(row['Good(Blue)'], row['Normal(Green)'], row['Bad(Yellow)'], row['Very bad(Red)']) 
    for idx, row in items.iterrows()
}

stations_dict = {row['Station code']: row['Station name(district)'] for idx, row in stations.iterrows()}

# aantal rijen verminderen
measures = measurements.pivot_table(index=['Measurement date', 'Station code', 'Instrument status'], columns='Item code', values='Average value').reset_index()
measures.columns = measures.columns.rename('')

# Geef de cijfers een betekenis 
intrument_status = {
    0: 'Normal',
    1: 'Need for calibration',
    2: 'Abnormal',
    4: 'Power cut off',
    8: 'Under repair',
    9: 'Abnormal data',
}
measures['Instrument status'] = measures['Instrument status'].replace(intrument_status)
measures['Station code'] = measures['Station code'].replace(stations_dict)
measures = measures.rename(columns=items_dict)

# hernoemen van kolommen
measures = measures.rename(columns={
    'Measurement date': 'Date',
    'Station code': 'Station',
    'Instrument status': 'Status'
})

# levels toevoegen 
for pol, func in evaluators.items():
    measures[pol.split()[0] + ' Level'] = measures[pol].map(func)
    
# Casting
measures['Date'] = pd.to_datetime(measures['Date'])

# 
weekday_dict = {
    0:'Monday',
    1:'Tuesday',
    2:'Wednesday',
    3:'Thursday',
    4:'Friday',
    5:'Saturday',
    6:'Sunday'
}

measures['Month'] = measures['Date'].dt.month
measures['Year'] = measures['Date'].dt.year
measures['Hour'] = measures['Date'].dt.hour
measures['Day'] = measures['Date'].dt.weekday.replace(weekday_dict)

district_pol = measures.groupby(['Station']).mean().loc[:, 'SO2 (ppm)':'PM2.5 (mircrogram/m3)']
district_pol_norm = (district_pol - district_pol.mean()) / district_pol.std()
district_pol_norm.columns = list(map(lambda x: x.split(' ')[0],district_pol_norm.columns))

pollution_map = folium.Map(height=800, width=1250,location=[37.562600,127.024612], tiles='cartodbpositron', zoom_start=11)

# Voeg punten toe aan de map
for idx, row in stations.iterrows():
    Marker([row['Latitude'], row['Longitude']], popup=row['Station name(district)'],tooltip='Klik hier om de popup te zien',icon=folium.Icon(color='green', icon='ok-sign')).add_to(pollution_map)

# Voeg choropleth toe
tile_layer = Choropleth(
    geo_data=district_borders,
    data=district_pol_norm.mean(axis=1), 
    key_on="feature.properties.name_eng", 
    fill_color='YlGn', 
    legend_name='Algemene luchtvervuiling in Seoul per regio'
).add_to(pollution_map)

folium.LayerControl().add_to(pollution_map)

# display(pollution_map)


# ## Statistische analyse

fig_scatter_pollution = px.scatter(df_air, x=['CO','NO2','SO2','O3'], y='PM2.5', trendline='ols', 
                                   labels={'variable': 'Type:','value': 'Waarde [ppm]', 'PM2.5': 'Ultra fijn stof [µm/m3]'})

fig_scatter_pollution.update_layout(height=650, width=1425,title="<b>Verhouding fijnstof vs type uitstoot<b>",
    updatemenus=[
        dict(
            active=0,
            
            buttons=list([
                dict(label="All",
                     method="update",
                     args=[{"visible": [True]},
                           {"title": "<b>Verhouding fijnstof vs type uitstoot in Seoul<b>"}]),
                dict(label="CO",
                     method="update",
                     args=[{"visible": [True,True,False,False,False, False,False,False]},
                           {"title": "<b>Verhouding fijnstof vs type uitstoot in Seoul<b>"}]),
                dict(label="NO2",
                     method="update",
                     args=[{"visible": [False,False,True,True,False, False,False,False]},
                           {"title": "<b>Verhouding fijnstof vs type uitstoot in Seoul<b>",
                            }]),
                dict(label="SO2",
                     method="update",
                     args=[{"visible": [False,False,False,False,True, True,False,False]},
                           {"title": "<b>Verhouding fijnstof vs type uitstoot in Seoul<b>",
                            }]),
                dict(label="O3",
                     method="update",
                     args=[{"visible": [False,False,False,False,False, False,True,True]},
                           {"title": "<b>Verhouding fijnstof vs type uitstoot in Seoul<b>",
                            }]),]),
            ),
    ])
fig_scatter_pollution.add_annotation(x=0, y=150,
            text='Lucht kwaliteit Index= Moderate',
            showarrow=True,
            arrowhead=1)
fig_scatter_pollution.add_annotation(x=0, y=80,
            text='Lucht kwaliteit Index= Normal',
            showarrow=True,
            arrowhead=1)
fig_scatter_pollution.add_hrect(y0=80, y1=150, line_width=0, fillcolor="yellow", opacity=0.2)
fig_scatter_pollution.add_hrect(y0=30, y1=80, line_width=0, fillcolor="green", opacity=0.2)
# fig_scatter_pollution.show()


# # Streamlit

st.set_page_config(layout="wide")

#Layout
st.header('Luchtvervuiling in Zuid-Korea')
st.text('Auteurs: Anna de Geeter, Chayenna Maas')

col1, col2, col3= st.columns(3)

with col1:
    st.write(histo)
    st.write(fig_scatter_pollution)
    
with col2:
    st.write(scatter)
    st.write(fig_line)
    
with col3:
    st.write(bar)
    folium_static(pollution_map)


st.subheader('Bronnen')
st.text('https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_topo_simple.json')
st.text('https://www.kaggle.com/datasets/bappekim/air-pollution-in-seoul')


# # Dash

# from dash import Dash, html, dcc

# pollution_map.save('index.html')

# app = dash.Dash(__name__)

# app.layout = html.Div(children=[
#     html.H1(children='Lucht vervuilingin Korea'),
#      html.Div(children='''
#         Auteurs: Anna de Geeter, Chayenna Maas.
#     '''),
#     dbc.Row([dbc.Col(html.Div(dcc.Graph(figure=bar, style={'width': '55vh'}))),
#                  dbc.Col(html.Div(dcc.Graph(figure=histo, style={'width': '55vh'})))]),
#     dbc.Row([(html.Div(dcc.Graph(figure=scatter, style={'width': '90%'})))]),
#     dbc.Row([dcc.Graph(figure=fig_line)]),
#     dbc.Row([dcc.Graph(figure=fig_scatter_pollution)]),
#     dbc.Row([html.Iframe(srcDoc = open('index.html', 'r').read(),style={'width': '120vh', 'height': '90vh'})]),
    
#     ])
    
# if __name__ == '_main_':
#     app.run_server(debug=False)

