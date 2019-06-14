# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:24:47 2018

@author: Jonathan
"""
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from queue import Queue
from threading import Thread, Lock
from fuzzywuzzy import fuzz

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import colorlover as cl
from sklearn.cluster import KMeans
# Data Cleaning Functions


def remove_annotation(text):
    """
    Checks for and removes annotations such as (2017 est)
    
    Parameters:
    ----------------
    text: string
        String of data to format
         
    Returns:    
    ----------------
    formatted: string
        Text without annotations
    
    """  
    match1 = re.search(r'\(\d.*', text)
    if match1:
        return text.replace(match1.group(), '').rstrip()
    else:
        return text
    
def get_unit(text):
    """
    Detects and returns unit of string, excludes numbers
    
    Parameters:
    ----------------
    text: string
        Text to be parsed
        
    Returns:    
    ----------------
     unit: string
        String representing the unit
    
    """
    match = re.search(r'([a-z]*)(illion)', text)    
    if match: 
        text = text.replace(match.group(),'').rstrip()
        
    if re.search(r'\$', text):
        unit = '$'        
    else:
        try:        
            match = re.search(r'[\d.,]*', text)
            if match:
                unit = text.replace(match.group(), '').lstrip()        
            else:
                unit = ''
        except:
            print(text)
            unit=''
    return unit


def remove_unit(text):
    """
    Checks for, returns, and removes unit from string
    
    Parameters:
    ----------------
    text: string
        Text to be modified
        
    Returns:    
    ----------------
    unit: string
        Bit of text that was removed
    text: 
        Text stripped of unit and any unwanted whitespace
    
    """
    dollar = re.search(r'\$', text)
    if dollar:
        
        text = text.replace('$', '')
    match1 = re.search(r'[\d.,]*', text)
    if match1:
        text = text.replace(match1.group(), '').rstrip()
    return text

def convert_to_num(text):
    """
    Converts unit free text into numbers
    
    Parameters:
    ----------------
    text: string
        description
        
    Returns:    
    ----------------
    num: float
         Data converted to numerical form
    
    """
    factor = 1
    unit = re.search(r'([a-z]*)(illion)', text)
    if unit:
        if unit.group(1)=='m':
            factor = 1e6
        elif unit.group(1)=='b':
            factor = 1e9
        elif unit.group(1)=='tr':
            factor = 1e12
        text = text.replace(unit.group(),'').rstrip()
    
    found =  re.search(r'-?[\d,]+[\.]?\d*', text.replace('$', ''))
    
    if found:        
        try:
            num= float(re.search(r'-?[\d,]+[\.]?\d*', text).group().replace(',',''))*factor
        except:
            num = text
            print('Had trouble converting %s' % text)
    else:
        num = 0
    return num

# Datascraping Functions
    
def get_soup(url, sleep_length=5):
    """
    Return a BeautifulSoup object of the url
    
    Parameters:
    ----------------
    url: string
        Website address of a country profile in the CIA world factbook
    sleep_length: int
        Length of pause in seconds before trying another request
        
    Returns:    
    ----------------
    html_soup: BeautifulSoup object
         description
    raw_text: string
        Raw html text of the webpage
    
    """ 
    while True:
        try:
            r = requests.get(url)
            break
        except requests.exceptions.RequestExceptions as e:
            print(e)
            sleep(sleep_length)
    
    raw_text = r.text
    html_soup = BeautifulSoup(raw_text, 'html.parser')
    return html_soup, raw_text

def sites_list(banned=[]):
    """Return a dict of CIA World Factbook country profile urls"""   
    
    # Search for country two-letter abbreviations
    url = 'https://www.cia.gov/library/publications/resources/the-world-factbook/'
    url_base = 'https://www.cia.gov/library/publications/resources/the-world-factbook/geos/%s.html'
    html_soup, text = get_soup(url)
    country_sites = {}
    for found in html_soup.find_all('option'):
        match = re.search(r'([a-z][a-z])\.(html)', str(found))
        if match:
            name = str(found.string).rstrip().lstrip()
            if not name in banned:
                country_sites[name] = url_base % match.group(1)
            
    return country_sites

def site_crawl(country_site, sleep_length=0):
    """
    Test retrieval of site information by printing out country name.
    
    Parameters
    -----------
    banned : list
        Names of countries to exclude from the sites to crawl
    sleep_length : int
        Number of seconds to pause before retrying a web page request
    """
    

    html_soup, text = get_soup(country_site, sleep_length)
    print(html_soup.find('span', {'class': 'region_name1 countryName '}).string)

    

def land_use(html_soup):
    """
    Creates a hierarchical DataFrame for land use data from the CIA world factbook
    
    Parameters:
    ----------------
    html_soup: BeautifulSoup object
        Object to navigate and find data from
    
    Returns:    
    ----------------
    Fields : list
        List of fields to be used in MultiIndex
    Subfields1 : list
        List of subfields to be used in MultiIndex
    Units : list
        List of units for the data to be used in MultiIndex
    Data : list
        List of numerical percents to be used in MultiIndex
    """
    field = 'Land use'
    Fields, Subfields, Units, Data = [], [], [], []
    found = html_soup.find('a', href=re.compile('.*#\d+'), string=field).find_next('div')
    next_num = found.find_next('span', class_="subfield-number")
    while True:
        name = next_num.find_previous('span', class_="subfield-name").string
        num = next_num.string
        Fields.append('Land use')
        Subfields.append(name)
        Units.append('%')
        Data.append(convert_to_num(num))
        next_num = next_num.find_next('span', class_="subfield-number")        
        if name== 'other:':
            break

    #columns = pd.MultiIndex.from_arrays(arrays=[Fields, Subfields1], names=['Field', 'Subfield'])
    #land_df = pd.DataFrame(np.array([Data1]), columns=columns)
        
    return Fields, Subfields, Units, Data


def elevation(html_soup):
    """
    Creates a hierarchical DataFrame for land use data from the CIA world factbook
        
    Parameters:
    ----------------
    html_soup: BeautifulSoup object
        Object to navigate and find data from
        
    Returns:    
    ----------------
    Fields: list
         Strings representing the Field level in the MultiIndex
    Subfields: list
         Strings representing the Subfields level in the MultiIndex
    Units: list
        Strings representing the Units level in the MultiIndex
    Data: list
         Numerical data    
    """      
    
    field = 'Elevation'
    Fields = ['Elevation', 'Elevation', 'Elevation']
    Subfields = ['mean elevation:', 'lowest point:', 'highest point:']
    Units = ['m','m','m']
    Data = []
    
    name = html_soup.find('span', {'class': ['region_name1', 'countryName']}).string
    
    div_field = html_soup.find('a', href=re.compile('.*#\d+'), string=field).find_next('div')
    mean = div_field.find_next('div')
    lowest = mean.find_next('div')
    hi = lowest.find_next('div')
    try:
        Data = [convert_to_num(str(i)) for i in [mean, lowest, hi]]
                
    except:
        Data = [None, None, None]
        print('Problem with elevation data for %s' % name)
    return Fields, Subfields, Units, Data


def age_structure(html_soup):
    """
    Creates a hierarchical DataFrame for Age structure data from the CIA world factbook
    
    Parameters:
    ----------------
    html_soup: BeautifulSoup object
        Object to navigate and find data from
        
    Returns:    
    ----------------
    Fields: list
         Strings representing the Field level in the MultiIndex
    Subfields: list
         Strings representing the Subfields level in the MultiIndex
    Units: list
        Strings representing the Units level in the MultiIndex
    Data: list
         Numerical data    
    """  
    
    Fields = []
    Subfields = []
    Units = []
    Data = []
    field = 'Age structure'    
    div_field = html_soup.find('a', href=re.compile('.*#\d+'), string=field).find_next('div')
    div_categories = div_field.find_all('div', class_='numeric')
    for i in div_categories:
        name = i.find('span', class_='subfield-name').string
        num = i.find('span', class_='subfield-number').string
        note = i.find('span', class_='subfield-note').string
        match = re.search(r'(\w+)\s([\d,]+)[\s/]*(\w+)\s([\d,]+)', note)
        Fields.extend([field, field, field])
        Subfields.extend(['total' +' '+ name, match.group(1)+' '+name, match.group(3)+' '+name])
        Units.extend(['percent', 'individuals', 'individuals'])
        Data.extend([convert_to_num(i) for i in [num, match.group(2), match.group(4)]])
    return Fields, Subfields, Units, Data

def improved_unimproved(field, html_soup):
    """
    Creates a hierarchical DataFrame for Drinking water source and Sanitation facility access 
    data from the CIA world factbook since they have similar data formats
    
    Parameters:
    ----------------
    field : string
        String 
    html_soup: BeautifulSoup object
        Object to navigate and find data from
    field_keys: dict
        Dictionary of field:number pairs in the CIA world factbook            
        
    Returns:    
    ----------------
    Fields: list
         Strings representing the Field level in the MultiIndex
    Subfields: list
         Strings representing the Subfields level in the MultiIndex
    Units: list
        Strings representing the Units level in the MultiIndex
    Data: list
         Numerical data    
    """  
    
    Fields = []
    Subfields = []
    Units = []
    Data = []
    div_field = html_soup.find('a', href=re.compile('.*#\d+'), string=field).find_next('div')
    
    div_field = html_soup.find('a', href=re.compile('.*#\d+'), string='Drinking water source').find_next('div')
    for i in div_field.find_all('span', class_='subfield-number'):
        group = i.find_previous('span', class_='subfield-group').string
        name = i.find_previous('span', class_='subfield-name').string
        Fields.append(field)
        Subfields.append(group+' '+name)
        Data.append(convert_to_num(i.string))
        Units.append(get_unit(i.string))
    return Fields, Subfields, Units, Data
        
def find_name(html_soup):
    """Returns name of country"""
    return html_soup.find('span', {'class': ['region_name1', 'countryName']}).string

def find_field(field, html_soup):
    """
    Finds field data for conventional cases and returns lists to make a hierarchical DataFrame.
    Excludes Land use and Elevation
    
    Parameters:
    ----------------
    field: string
        Exact string to match field on webpage
    
    html_soup: BeautifulSoup object
        Object to navigate and find data from
    field_keys: dict
        Dictionary of field:number pairs in the CIA world factbook        
        
    Returns:    
    ----------------
    Fields: list
         Strings representing the Field level in the MultiIndex
    Subfields: list
         Strings representing the Subfields level in the MultiIndex
    Units: list
        Strings representing the Units level in the MultiIndex
    Data: list
         Numerical data
    
    """ 
    
    
        
    Fields, Subfields, Units, Data = [], [], [], []
    
    # Check if field is present
    link_tag = html_soup.find('a', href=re.compile('.*#\d+'), string=field)
    if link_tag:
        
        # check if field is a special case
        if field == 'Land use':
            return land_use(html_soup)
        elif field == 'Elevation':
            return elevation(html_soup)    
        elif field == 'Age structure':
            return age_structure(html_soup)
        elif field in ['Drinking water source', 'Sanitation facility access']:
            return improved_unimproved(field, html_soup)
        
        
        div_field =  link_tag.find_next('div')
                    
        for div_category_data in div_field.find_all('div', class_=re.compile('category_data')):
            
            name = div_category_data.find('span', class_="subfield-name")
            num = div_category_data.find('span', class_="subfield-number")
                                    
            # Check for scenario 2
            if 'text' in div_category_data['class']:
                continue
            
            # Check for scenarios 3 and 4    
            elif name==None or {'note','historic'} <= set(div_category_data['class']):
                if num:                    
                    Fields.append(field)
                    Units.append(get_unit(num.string))
                    Data.append(convert_to_num(num.string))                    
                    Subfields.append('')                    
                break        
            
            # Remaining case is situation 1
            else:
                if num:
                    Fields.append(field)
                    Units.append(get_unit(num.string))
                    Data.append(convert_to_num(num.string))
                    Subfields.append(name.string)
                else:
                    continue
    return Fields, Subfields, Units, Data

def create_dataframe(Fields, Subfields, Units, Data):
    """
    Creates a hierarchical data frame from the inputted lists
    
    Parameters:
    ----------------
    Fields: list
         Strings representing the Field level in the MultiIndex
    Subfields: list
         Strings representing the Subfields level in the MultiIndex
    Units: list
        Strings representing the Units level in the MultiIndex
    Data: list
         Numerical data
        
    Returns:    
    ----------------
    hier_df: dataframe object
         Pandas MultiIndex dataframe object
    
    """  
    columns = pd.MultiIndex.from_arrays(arrays=[Fields, Subfields, Units], names=['Field', 'Subfield', 'Units'])
    hier_df = pd.DataFrame([Data], columns=columns)
    return hier_df


    
def country_profile(name, url, fields, missing_fields, sleep_length=5):
    """
    Create a table of data given a CIA world factbook country profile and fields of interest.
    
    Parameters:
    ----------------
    name: string
        Name of country on the options menu           
    url: string
        Website address of a country profile in the CIA world factbook
    fields: list
        List of strings corresponding to the Fields of information referenced in:
        https://www.cia.gov/library/publications/the-world-factbook/docs/profileguide.html    
    missing_fields: defaultdict
        defaultdict with country: list of missing fields to skip unnecessary searches
    sleep_length: int
        Length of pause in seconds before trying another request
    
    Returns:    
    ----------------
     country_df: pandas DataFrame object
         Table containing the string contents of selected fields on the site
    """ 
    
    html_soup, text = get_soup(url, sleep_length)
    Fields_ = ['Name']
    Subfields_ = ['']
    Units_ = ['']
    Data_ = [str(html_soup.find('span', {'class': ['region_name1', 'countryName']}).string)]
        
    for field in fields:
        if not html_soup.find('a', href=re.compile('.*#\d+'), string=field):
            continue
        Fields, Subfields, Units, Data = find_field(field, html_soup)        
        Fields_.extend(Fields)
        Subfields_.extend(Subfields)
        Data_.extend(Data)
        Units_.extend(Units)
    
    return Fields_, Subfields_, Units_, Data_

    
def world_table(fields, banned=[], sleep_length=5):
    """
    Creates a table version of selected CIA world factbook data. Requires numpy as np, 
    pandas as pd, BeautifulSoup, request, and re libraries.
    
    Parameters:
    ----------------
    fields: list
        List of exact field names to match
    banned: list
        List of countries to skip
    sleep_length: int
        Length of pause in seconds before trying another request
        
    Returns:    
    ----------------
     table: pandas dataframe
         MultiIndex data frame with strings representing data in the specified fields
         
    """ 
    # Create a list of country profile urls
    country_sites = sites_list(banned)
    site_countries = {value: key for key, value in country_sites.items()}
    missing_fields = not_found(fields)
    frames = []
    for name, url in country_sites.items():
        try:
            html_soup, text = get_soup(url, sleep_length)
            frames.append(country_profile(name, url, fields, missing_fields, sleep_length))
        except AttributeError as err:
            print('%s was skipped due to %s.' % (site_countries[url], err))
        except IndexError as err:
            print('%s was skipped due to %s.' % (site_countries[url], err))
            
    #return pd.concat(frames)
    return frames

def world_table_threaded(fields, banned=[], sleep_length=5):
    """
    Creates a threaded table version of selected CIA world factbook data. Requires numpy as np, 
    pandas as pd, BeautifulSoup, request, threading and re libraries.
    
    Parameters:
    ----------------
    fields: list
        List of exact field names to match
    banned: list
        List of countries to skip
    sleep_length: int
        Length of pause in seconds before trying another request
        
    Returns:    
    ----------------
    frames: list 
         List of pandas MultiIndex dataframes 
         
    """ 
    frames = []
    NUM_WORKERS = 4
    country_sites = sites_list(banned)
    site_countries = {country_sites[country]:country for country in country_sites}
    
    def worker():
        
        while True:
            # Constantly check the queue for addresses
            url = task_queue.get()
            name = site_countries[url]
            try:
                html_soup, text = get_soup(url, sleep_length)
                frames.append(create_dataframe(*country_profile(name, url, fields, sleep_length)))
                print(name, ' done')
            except AttributeError as err:
                print('%s was skipped due to %s.' % (site_countries[url], err))
            except IndexError as err:
                print('%s was skipped due to %s.' % (site_countries[url], err))
                
            # Mark task as done
            task_queue.task_done()

    # Create threads and Q
    task_queue = Queue()
    # Add website to task queue
    [task_queue.put(country_sites[country]) for country in country_sites]

    threads = [Thread(target=worker) for _ in range(NUM_WORKERS)]


    # Start all workers
    [thread.start() for thread in threads]

    # Wait for all the tasks in the queue to be processed
    task_queue.join()
    return frames

# Database Functions

def sql_create_title(text):
    """
    Generate SQL compatible title from raw text
    
    Parameters:
    -------------
    text : string
        The candidate title name to be converted
    
    Returns:
    -------------
    title : string
        SQL friendly table title
    
    """    
    
    removal = [r'\s\(\+\)', r'\s\(\-\)', r'[(),:]', r'\'',r'%']
    replacement = ['/', '\s\-\s', '\s', '-']
    
    # No starting numeric character
    if re.search(r'\d', text[0]):
        text = '_' + text
    # Special character removal
    for i in removal:
        text = re.sub(i, '', text)
    for i in replacement:
        text = re.sub(i,'_', text)
    # White space removal
    text = text.lstrip().rstrip()        
    return text


def sql_add_column(table, column, data_type):
    """
    Generates SQL text needed to add new table columns
    
    Parameters:
    ---------------
    table: string
        The name that will be used as the name of the table
    column: string
        Name of the new column
    data_type: string
        SQLite data type, ie real, text, blob
        
    Returns:    
    ----------------
    sql: string
        String to use in connection.execute(sql)
       
    """  
    sql = f"""ALTER TABLE {table} ADD {column} real;"""
    return sql

def sql_create_table(field, columns):
    """
    Generate the SQL text needed to create a table representing the fields
    
    Parameters:
    ----------------
    field: string
        The field name that will be used as the name of the table
    columns: list of strings
        A list of strings that will be the subfields or subcategories and the names of the columns
        and excludes country
        
    Returns:    
    ----------------
    sql: string
         String to use in connection.execute(sql)
    
    """  

    sql = f"""CREATE TABLE {field} (\n country text PRIMARY KEY,            
        """
    for i in columns[1:]:
        sql+=f"\n \t {i} real," 
    sql = sql[:-1]+");"
    return sql

def sql_table_columns(table_title, conn):
    """
    Generate list of current table columns
    
    Parameters:
    ----------------
    table_title: string
        The name of the table
    
    conn : sqlite3 connection object
    
    Returns:    
    ----------------
    current_columns: list of strings
         List of table columns    
    """  
    
    sql = f"""PRAGMA TABLE_INFO({table_title})"""
    df = pd.read_sql_query(sql, conn)
    columns = df['name']
    return columns



def sql_update_table(table_title, columns, data):
    """
    Generates SQL compatible string that can be fed into the execute method with
    the data
    
    Parameters:
    ---------------
    table_title: string
        The name of the table
    columns: list of strings
        A list of strings that will be the names of the table columns for each corresponding entry in data
    data : list or tuple of floats and strings
        A list of floats representing the data to be inserted into the table, with entries corresponding to columns
        
    Returns:    
    ----------------
    sql: string
         String to use in connection.execute(sql)
    
    """  
    col = ""
    val = ""
    for i, j in zip(columns, data):
        col+=f'{i},'   
        val+=f"""?,"""   
        sql = f"""INSERT INTO {table_title} ({col[:-1]}) VALUES ({val[:-1]});"""    
    return sql
 
def sql_fill_table(table_title, data):
    """
    Generates SQL compatible string that can be fed into the execute method with
    the data
    
    Parameters:
    ---------------
    table_title: string
        The name of the table
    data : list of tuples of floats
        A list of tuples of floats representing the data to be inserted into the table, with entries corresponding to columns
        
    Returns:    
    ----------------
    sql: string
         String to use in connection.executemany(sql)
    
    """
    
    sql = f"""INSERT INTO {table_title} VALUES ("""
    
    for i in data[0]:
        sql+='?,'
    sql = sql[:-1] +');'        
    return sql      

def sql_table_contents(table_title,conn):
    """
    Returns a SQL table as a dataframe
    
    Parameters:
    ----------------
    table_title: string
        The name of the table
    
    conn : sqlite3 connection object
    
    Returns:    
    ----------------
    df: pandas DataFrame object
         Dataframe version of table 
    """      
    df = pd.read_sql_query(f'SELECT * FROM {table_title};', conn)
    return df
# Name Matching Function
        
def word_match(words, aliases, cutoff=75):
    """
    Chooses aliases for names. Requires
    
    from fuzzywuzzy import fuzz
    
    Parameters:
    ----------------
    words: list
        List of names to assign aliases to
    aliases: type
        description
    cutoff: int
        Minimum score for string similarity to qualify an alias
    
    Returns:    
    ----------------
    name_aliases: dict
         Dictionary with name:alias pairs
    
    """  
    words_aliases = dict()
    for i in words:
        matches = [fuzz.token_set_ratio(i,j) for j in aliases]
        if max(matches) >= cutoff:
            words_aliases[i] = aliases[np.argmax(matches)]
        else:
            words_aliases[i] = i
    return words_aliases

# Visualization Functions
    
def plot_global_chloropleth(df, cbar_title='', source_text=''):
    """
    Wrapper function that takes in a dataframe of country names and a quantitative field and then plots
    a Viridis chloropleth with many default settings of the plotly chloropleth example.
    
    Parameters:
    ----------------
    df: pandas DataFrame object
        First column must be country names and the second a quantitative field
    cbar_title: string
        Title to put on the colorbar
    source_text: string
        Annotation to give to the bottom of the map
    """
    data = [go.Choropleth(
        locations = df[df.columns[0]],
        locationmode = 'country names',
        z = df[df.columns[1]],
        colorscale='Viridis',
        autocolorscale = False,
        reversescale = True,
        marker = go.choropleth.Marker(
            line = go.choropleth.marker.Line(
                color = 'rgb(180,180,180)',
                width = 0.5
            )),
        colorbar = go.choropleth.ColorBar(
            title = cbar_title),
    )]

    layout = go.Layout(
        title = go.layout.Title(
            text = df.columns[1]
        ),
        geo = go.layout.Geo(
            showframe = False,
            showcoastlines = False,
            projection = go.layout.geo.Projection(
                type = 'equirectangular'
            )
        ),
        annotations = [go.layout.Annotation(
            x = 0.55,
            y = 0.1,
            xref = 'paper',
            yref = 'paper',
            text = source_text,
            showarrow = False
        )]
    )

    fig = go.Figure(data = data, layout = layout)
    iplot(fig, filename = 'd3-world-map')

def create_k_clusters(k, X, text=''):
    """
    Creates a list of scatter objects for clustering visualization
    
    Parameters:
    ----------------
    k: int
        Number of clusters to fit data to
    X: numpy array
        n x 2 array with coordinates of each point   
    text: string or array of strings
        Text which will pop up when hovering over data points
    Returns:    
    ----------------
    data: list of plotly.graph_objs.Scatter objects
         Contains Scatter objects for two dimensional plotting    
    """
    data = []
    colors_k_bins = cl.to_rgb(cl.interp(cl.scales['11']['qual']['Paired'], k))        
    class_labels_k = KMeans(k).fit_predict(X)    
    for i,j in enumerate(colors_k_bins):
        
        data.append(go.Scatter(x = X[class_labels_k==i,0],
                               y = X[class_labels_k==i,1],
                               text = text,
                               mode = 'markers',
                               marker=go.scatter.Marker(color=j)                             
                               )
                   )        
    return data

def create_clustering_buttons(k):
    """
        
    Parameters:
    ----------------
    k: int
        Maximum clusters
        
    Returns:    
    ----------------
    updatemenus: list
         List of buttons to input into the plot layout  
    """ 
    buttons = []
    
    for i in range(k):
        # sets which clusters to be visible for each clustering scheme
        n_clusters= i+1
        args=np.array([False for i in range(int((k+1)*k/2))])
        args[int(n_clusters*(n_clusters+1)/2-n_clusters):int(n_clusters*(n_clusters+1)/2)]=True
        if i == 0:
            label = '1 Cluster'
        else:
            label = '%s Clusters' % n_clusters
        buttons.append(dict(method='update',
                            args=[{'visible': list(args)}],
                            label= label))        
    return buttons
    
def plot_k_clusters(k, X, title, text='', xlabel='', ylabel=''):
    """
    Plots k clusters in plotly, specifying highlighted text
    
    Parameters:
    ----------------
    k: int
        Number of clusters to fit data to
    X: numpy array
        n x 2 array with coordinates of each point
    title: string
        Title of scatter plot
    text: string
        Text to show up upon hovering over data points
    xlabel: string
        Label for x axis
    ylabel: string
        Label for y axis       
    """  
    data = create_k_clusters(k, X)
    
    layout = dict(title = title,
              xaxis= dict(title= xlabel),
              yaxis= dict(title= ylabel)
             )
    fig = dict(data = data, layout = layout)
    iplot(fig)    

def plot_variable_clusters(k_max, X, title, text='', xlabel='', ylabel=''):
    """
    Plots 1 to k clusters on same data organized by buttons
    
    Parameters:
    ----------------
    k: int
        Number of clusters to fit data to
    X: numpy array
        n x 2 array with coordinates of each point
    title: string
        Title of scatter plot
    text: string
        Text to show up upon hovering over data points
    xlabel: string
        Label for x axis
    ylabel: string
        Label for y axis       
    """  
    updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=create_clustering_buttons(k_max)
        )])
    
    traces_k_clusters = dict()
    
    data = []
    for i in range(k_max):
        k_clusters = i+1
        data.extend(create_k_clusters(k_clusters, X))
    layout = dict(title=title, updatemenus=updatemenus, showlegend=True)

    fig = dict(data=data, layout=layout)
    iplot(fig)
 