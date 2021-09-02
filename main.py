import datetime
import folium
import geopandas as gpd
import geopy
import networkx as nx
import osmnx as ox
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import time

from folium.features import DivIcon
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from PIL import Image
from streamlit_folium import folium_static

st.set_page_config(
    page_title="California Earthquake Safe Path",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    """
    <style>
    .css-17eq0hr {
        background-color:  #aed6f1 !important;
    }
    .st-bm {
        margin-left: 7% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



graph = nx.read_gpickle('2_all_graph_all_risk_added.pickle')
nodes, edges = ox.graph_to_gdfs(graph)

HtmlFile = open("heatmap.html", 'r', encoding='utf-8')
heatmap_html = HtmlFile.read() 

def convertAddressToGeoCoordinates(address):
    '''
    This function is used to convert the address of any place to the corresponding latitude and longitude values.

    Parameters
    ----------
    address: str

    Returns
    -------
    coordinates: tuple 
                coordinates[0] = latitude 
                coordinates[1] = longitude

    '''
    geolocator = Nominatim(user_agent="Nominatim")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=5)
    address_latlon = geocode(address)
    coordinates = (address_latlon.latitude, address_latlon.longitude)
    return coordinates

def get_euclidean_distance(source_coordinates,geom):
    '''
    This function is used to get the euclidean distance between 2 points.

    Parameters
    ----------
    source_coordinates: tuple
                        source_coordinates[0] = latitude
                        source_coordinates[1] = longitude
    geom: point

    Returns
    --------
    float: Distance between the 2 points
    ''' 
    return ox.distance.euclidean_dist_vec(source_coordinates[0],source_coordinates[1],geom.bounds[1],geom.bounds[0])


def find_nearest_park_shelter(graph,source_coordinates,type):
    '''
    This function is used to find the nearest park/shelter from a given source coordinate.

    Parameters
    ----------
    graph: NetworkX Graph
    source_coordinates: tuple
                        source_coordinates[0] = latitude
                        source_coordinates[1] = longitude
    type: str
        type can be 'park' or 'shelter'

    Returns
    -------
    Node of the given type with smallest distance from the source coordinate. 
    ''' 
    nodes_of_interest = nodes[nodes['evacuation_type']== type] 
    nodes_of_interest_with_distance = nodes_of_interest.geometry.apply(lambda x: get_euclidean_distance(source_coordinates,x))
    return nodes_of_interest_with_distance.sort_values().index.values[0]

    # global cache to speed up dist(heuristic) computation
    target_x = None
    target_y = None

def dist(a, b):
    global target_x
    global target_y
    #(x1, y1) = a
    #(x2, y2) = b
    node_a = nodes.at[a, 'geometry']
    x1 = node_a.x
    y1 = node_a.y

    x2 = target_x
    y2 = target_y

    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    #return abs(x1 - x2) + abs(y1 - y2)


def findPath(graph, source_coordinates, destination_coordinates, choice_of_destination, heuristic=None):
    '''
    This function calculates the best path from soucre to destination in the given graph based on the risk column.

    Parameters
    ----------
    graph: NetworkX Graph
    source_coordinates: tuple
                        source_coordinates[0] = latitude of source
                        source_coordinates[1] = longitude of source
    destination_coordinates: tuple
                        destination_coordinates[0] = latitude of source
                        destination_coordinates[1] = longitude od source
    risk_column_name: str
                    risk_column_name is the name of a column in edges. The path is calculated based on the values of this column

    Returns
    -------
    list[list]
    list[0] contains path. The path consists of osmid of the nodes in the path.
    '''


    global target_x
    global target_y


    start_time = time.time()
    # find the nearest node to the source coordinats
    source_node, source_dist = ox.get_nearest_node(graph,source_coordinates,return_dist=True)
    # find the destination if not given
    if choice_of_destination == 1: # park
        destination_node = find_nearest_park_shelter(graph,source_coordinates,'park')
    elif choice_of_destination == 2: # shelter
        destination_node = find_nearest_park_shelter(graph,source_coordinates,'shelter')
    else:
        destination_node, destination_dist = ox.get_nearest_node(graph,destination_coordinates,return_dist=True)

    target_node = nodes.at[destination_node, 'geometry']
    target_x = target_node.x
    target_y = target_node.y
    path = nx.astar_path(G=graph, source=source_node, target=destination_node, heuristic = heuristic, weight='combined_risk')
    A_time = (time.time() - start_time)
    print('Destination node: ',destination_node)
    print("TOTAL TIME = ",A_time)

    #calculating the length and risk associated with the path
    nxg = nx.Graph(graph)
    path_length = sum(nxg[u][v].get('length') for u, v in zip(path[:-1], path[1:]))
    path_risk = sum(nxg[u][v].get('combined_risk') for u, v in zip(path[:-1], path[1:]))
    print("LENGTH OF THE ROUTE = ",path_length)
    print("RISK OF THE ROUTE = ",path_risk)
    if choice_of_destination in (1,2):
        destination_coordinates = (nodes.loc[destination_node].geometry.y, nodes.loc[destination_node].geometry.x)

    #plot_path(path, source_coordinates, destination_coordinates)

    return [path, path_length, path_risk, source_coordinates, destination_coordinates]


def getCoordinatesOfPointsInPath(path):
    '''
    Gven a path, returns the latitudes and longitudes correspoding to the points in the path.

    Parameters
    ----------
    path: list of osmid of nodes in the path

    Returns
    -------
    A list with two items.
    The first item is a list containing all the latitude values.
    The second item is a list containing all the longitude values.
    '''
    coords = []
    long = [] 
    lat = []  

    for i in path:
        point = nodes.loc[i]
        long.append(point['x'])
        lat.append(point['y'])
        coords.append([point['y'], point['x']])
    return [lat, long, coords]



st.sidebar.markdown('<h1 style="margin-left:8%; color:#1a5276">California Earthquake Safe Path </h1>', unsafe_allow_html=True)

add_selectbox = st.sidebar.radio(
    "",
    ("Find Path", "Maps")
)


if add_selectbox == 'Find Path':
    col1, col2 = st.columns([5, 2])
    source_address = col1.text_input('Current Location:') 
    risk_type = col2.selectbox('Select Destination Type', ('Nearest Park and Shelter','Nearest Park', 'Nearest Shelter', 'Custom Destination'))
    if risk_type == 'Custom Destination':
        destination_address = col1.text_input('Custom Destination:') 
    
    if st.button('Search'):
        try:
            if source_address.strip() == '':
                st.write('Please Input Source Address')
            elif risk_type == 'Custom Destination' and destination_address.strip() == '':
                st.write('Please Input Source Address')
            else:
                destination_coordinates = ''
                source_coordinates = convertAddressToGeoCoordinates(source_address)
                if risk_type == 'Nearest Park and Shelter':
                    choice_of_destination = 1
                    path_info_1 = findPath(graph, source_coordinates,destination_coordinates,choice_of_destination)
                    
                    m = folium.Map(location=[(path_info_1[3][0]+path_info_1[4][0])/2, (path_info_1[3][1]+path_info_1[4][1])/2],
                        zoom_start = 13, tiles='cartodbpositron')


                    lat, long, coords = getCoordinatesOfPointsInPath(path_info_1[0])
                    
                    folium.Marker([coords[0][0], coords[0][1]], icon=folium.Icon(color="green", prefix='fa')).add_to(m)
                    
                    folium.PolyLine(coords, popup='<b>Path of Vehicle_1</b>',
                                                        color='red',
                                                        weight=5).add_to(m)


                    folium.Marker([coords[len(coords)-1][0], coords[len(coords)-1][1]], 
                                popup='Park',
                                icon=folium.Icon(color="red", icon="tree", prefix='fa')
                                ).add_to(m)

                    choice_of_destination = 2
                    path_info_2 = findPath(graph, source_coordinates,destination_coordinates,choice_of_destination)

                    lat, long, coords_2 = getCoordinatesOfPointsInPath(path_info_2[0])
                    
                    
                    folium.PolyLine(coords_2, color='red', weight=5).add_to(m)
                    
                    folium.Marker([coords[len(coords_2)-1][0], coords[len(coords_2)-1][1]], 
                                popup='Shelter',
                                icon=folium.Icon(color="red", icon="home", prefix='fa')
                                ).add_to(m)
                    
                    col1, col2 = st.columns(2)
                    col1.markdown('<b>Details: </b> <br>', unsafe_allow_html=True)
                    col2.markdown('<b>&nbsp;</b><br>', unsafe_allow_html=True)
                    col1.markdown('<b>Park Path Length: </b>{}'.format(round(path_info_1[1])), unsafe_allow_html=True)
                    col1.markdown('<b>Park Total Risk: </b>{}'.format(round(path_info_1[2])), unsafe_allow_html=True)
                    col2.markdown('<b>Shelter Path Length: </b>{}'.format(round(path_info_2[1])), unsafe_allow_html=True)
                    col2.markdown('<b>Shelter Total Risk: </b>{}'.format(round(path_info_2[2])), unsafe_allow_html=True)

                else:
                    if risk_type == 'Custom Destination':
                        destination_coordinates = convertAddressToGeoCoordinates(destination_address)
                        choice_of_destination = 3
                    elif risk_type == 'Nearest Shelter':
                        choice_of_destination = 2
                    else:
                        destination_address = 'NA'
                        choice_of_destination = 1

                    
                    path_info = findPath(graph, source_coordinates,destination_coordinates,choice_of_destination)

                    st.markdown('<b>Path Length: </b>{}'.format(round(path_info[1])), unsafe_allow_html=True)
                    st.markdown('<b>Total Risk: </b>{}'.format(round(path_info[2])), unsafe_allow_html=True)

                    m = folium.Map(location=[(path_info[3][0]+path_info[4][0])/2, (path_info[3][1]+path_info[4][1])/2],
                        zoom_start = 13, tiles='cartodbpositron')

                    
                    lat, long, coords = getCoordinatesOfPointsInPath(path_info[0])
                    
                    folium.Marker([coords[0][0], coords[0][1]], icon=folium.Icon(color="green", prefix='fa')).add_to(m)
                        
                    folium.PolyLine(coords, popup='<b>Path of Vehicle_1</b>',
                                                        tooltip='Vehicle_1',
                                                        color='red',
                                                        weight=5).add_to(m)


                    folium.Marker([coords[len(coords)-1][0], coords[len(coords)-1][1]], 
                                icon=folium.Icon(color="red", icon="map-pin", prefix='fa')
                                ).add_to(m)


                folium_static(m, width=900)


        
        except Exception as e:
            st.write('Error:: {}'.format(str(e)))

else:
    risk_type = st.selectbox('Risk Factor', ('Combined Risk Factor','Building Risk Score', 'Distance Risk Score'))
    components.html(heatmap_html, height= 500, width=900)

