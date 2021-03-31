#! /usr/bin/python
#-*- coding: utf-8 -*-

## How to run:
## 1. make a folder and place the script in the folder
## 2. run pipreqs to create the requirements file; if not installed, use
##    $> pip install pipreqs
##    $> pipreqs .
##    $> pip install -r requirements.txt
## 3. Run the script

from __future__ import print_function
import datetime
import urllib.request
import shutil
import sys
import os
import re
import pandas
from pandas.core.common import flatten
import argparse
import matplotlib.pyplot as plt
import numpy as np

COVID_URI = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
csv_header = [
    'iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases',
    'new_cases_smoothed', 'total_deaths', 'new_deaths', 'new_deaths_smoothed',
    'total_cases_per_million', 'new_cases_per_million',
    'new_cases_smoothed_per_million', 'total_deaths_per_million',
    'new_deaths_per_million', 'new_deaths_smoothed_per_million',
    'reproduction_rate', 'icu_patients', 'icu_patients_per_million',
    'hosp_patients', 'hosp_patients_per_million', 'weekly_icu_admissions',
    'weekly_icu_admissions_per_million', 'weekly_hosp_admissions',
    'weekly_hosp_admissions_per_million', 'new_tests', 'total_tests',
    'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed',
    'new_tests_smoothed_per_thousand', 'positive_rate', 'tests_per_case',
    'tests_units', 'total_vaccinations', 'people_vaccinated',
    'people_fully_vaccinated', 'new_vaccinations', 'new_vaccinations_smoothed',
    'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
    'people_fully_vaccinated_per_hundred',
    'new_vaccinations_smoothed_per_million', 'stringency_index', 'population',
    'population_density', 'median_age', 'aged_65_older', 'aged_70_older',
    'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate',
    'diabetes_prevalence', 'female_smokers', 'male_smokers',
    'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy',
    'human_development_index'
]

static_columns = [
    'iso_code', 'continent', 'location', 'tests_units', 'population',
    'population_density', 'median_age', 'aged_65_older', 'aged_70_older',
    'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate',
    'diabetes_prevalence', 'female_smokers', 'male_smokers',
    'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy',
    'human_development_index'
]

geo_columns = ['iso_code', 'continent', 'location']


def get_csv(filename=None):
    """ Download the dataset in csv format
        If filename is provided, save the downloaded file as 'filename'.
    """
    if not filename:
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filename = 'covid_{}.csv'.format(timestamp_str)
    print('Downloading dataset file {:} to {:} ...'.format(COVID_URI, filename),
          end='')
    with urllib.request.urlopen(COVID_URI) as response, open(filename,
                                                             'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print(' done!')
    return filename


def load_dataframe(csv_file=None):
    if csv_file is None:
        csv_file = get_csv()
    else:
        if not os.path.isfile(csv_file):
            sys.exit(1)
    return pandas.read_csv(
        csv_file,
        parse_dates=[csv_header.index('date')],
        date_parser=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d'))


def countries_in_continent(df, continent):
    """
    """
    return list(
        flatten(df.loc[df['continent'] == continent,
                       ['location']].drop_duplicates(subset=['location'],
                                                     keep='last').values))


def get_owid_iso_codes(df):
    """ Returns [array(['OWID_AFR', 'Africa'], dtype=object), array(['OWID_ASI', 'Asia'], dtype=object), ...]
    """
    return list(df.loc[df['iso_code'].str.match('^OWID_[A-Z]+') == True,
                       ['iso_code', 'location']].drop_duplicates(
                           subset=['location'], keep='last').values)


def get_country_list(df):
    return list(df[geo_columns].drop_duplicates(subset=['location'],
                                                keep='last').fillna('').values)


def print_country_list(df):
    clst = get_country_list(df)
    print('{:10s} {:20s} {:25s}'.format(*geo_columns))
    for tpl3 in clst:
        if not re.match('^OWID_[A-Z]+', tpl3[0]):
            print('{:10s} {:20s} {:25s}'.format(tpl3[0], tpl3[1], tpl3[2]))
    for tpl3 in clst:
        if re.match('^OWID_[A-Z]+', tpl3[0]):
            print('{:10s} {:20s} {:25s}'.format(tpl3[0], tpl3[1], tpl3[2]))


def get_cols_by_location(df, **kwargs):
    """ Examples:
    get_cols_by_location(df, countries=['Greece'])
    get_cols_by_location(df, countries=['Greece'], columns=['tests_per_case', 'people_fully_vaccinated'])
    get_cols_by_location(df, continents=['Asia'], columns=['tests_per_case', 'people_fully_vaccinated'])
    get_cols_by_location(df, countries=['Greece', 'Spain', 'China'], columns=['tests_per_case', 'people_fully_vaccinated'])
    """
    ## users can request average owid values; these are marked in the dataset
    ## as (e.g. for arfica) OWID_AFR
    country_list = kwargs['countries'] if 'countries' in kwargs else []
    continent_list = kwargs['continents'] if 'continents' in kwargs else []
    owid_list = [
        i.upper()
        for i in country_list + continent_list
        if re.match('OWID_[A-Z]+', i.upper())
    ]
    columns = kwargs['columns'] if 'columns' in kwargs else []
    if columns == []:
        return df[df['location'].isin(country_list) |
                  df['continent'].isin(continent_list) |
                  df['iso_code'].isin(owid_list)]
    else:
        if columns != []:
            ## drop duplicate locations if we only want 'static' columns
            if len([x for x in columns if x in static_columns]) == len(columns):
                return df.loc[df['location'].isin(country_list) |
                              df['continent'].isin(continent_list) |
                              df['iso_code'].isin(owid_list),
                              columns].drop_duplicates(subset=['location'],
                                                       keep='last')
        return df.loc[df['location'].isin(country_list) |
                      df['continent'].isin(continent_list) |
                      df['iso_code'].isin(owid_list), columns]


def create_scatter(df, **kwargs):
    print('Creating scatter plot: {:}'.format(kwargs))
    assert ('columns' in kwargs and len(kwargs['columns']) == 2)
    x_axis = kwargs['columns'][0]
    ## at least for now, only allow date in x-axis
    if x_axis != 'date':
        print(
            '[ERROR] Only \'date\' is allowed as x-axis when creating a dynamic (scatter) plot.',
            file=sys.stderr)
        print('[ERROR] You requested x-axis to be \'{:}\''.format(x_axis),
              file=sys.stderr)
        print('[ERROR] Probable fix: set --x-axis=date', file=sys.stderr)
        raise RuntimeError('[ERROR] Invalid x-axis selection')
    y_axis = kwargs['columns'][1]
    plt.figure()
    legend = []
    if 'continents' in kwargs:
        kw_columns = kwargs['columns'] + ['location', 'continent']
        ndf = get_cols_by_location(df,
                                   continents=kwargs['continents'],
                                   columns=kw_columns)
        for continent in kwargs['continents']:
            for country in countries_in_continent(ndf, continent):
                xy = ndf.loc[ndf['location'] == country, [x_axis, y_axis]]
                plt.plot(xy[x_axis], xy[y_axis], marker='o')
                legend.append('{:}/{:}'.format(country, continent))
    if 'countries' in kwargs:
        ndf = get_cols_by_location(df,
                                   countries=kwargs['countries'],
                                   columns=kwargs['columns'] +
                                   ['location', 'iso_code'])
        for country in kwargs['countries']:
            ## treat potential 'special locations', aka 'OWID_AFR', etc ....
            key = 'location' if not re.match('^OWID_[A-Z]+',
                                             country.upper()) else 'iso_code'
            xy = ndf.loc[ndf[key] == country, [x_axis, y_axis]]
            plt.plot(xy[x_axis], xy[y_axis], marker='o')
            legend.append(country)
    plt.legend(legend)
    plt.title('Coivd-19 {:} Vs {:}; data source:\n{:}'.format(
        x_axis, y_axis, COVID_URI))
    plt.grid(True)
    plt.show()


def create_bar(df, **kwargs):
    print('Creating bar plot: {:}'.format(kwargs))
    frames = []
    if 'continents' in kwargs:
        frames.append(
            get_cols_by_location(df,
                                 continents=kwargs['continents'],
                                 columns=kwargs['columns']))
    if 'countries' in kwargs:
        frames.append(
            get_cols_by_location(df,
                                 countries=kwargs['countries'],
                                 columns=kwargs['columns']))
    ndf = pandas.concat(frames)
    labels = ndf['location']
    x = np.arange(len(labels))
    width = 0.35e0
    fig, ax = plt.subplots()
    bars = [
        ax.bar(x - width / 2, ndf[c].values, label=c)
        for c in kwargs['columns'][1:]
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    #for bar in bars:
    #  ax.bar_label(bar, padding=3)
    fig.tight_layout()
    plt.show()


class myFormatter(argparse.ArgumentDefaultsHelpFormatter,
                  argparse.RawTextHelpFormatter):
    pass


parser = argparse.ArgumentParser(formatter_class=myFormatter,
                                 description='',
                                 epilog=(''))

parser.add_argument('--csv-file',
                    metavar='INPUT_CSV_FILE',
                    dest='csv_file',
                    required=False,
                    default=None,
                    help='Use this data csv file instead of dwonloading.')
parser.add_argument(
    '--print-columns',
    dest='print_columns',
    action='store_true',
    help=
    'Print columns of dataframe, aka the recorded, raw data descrition of the dataset values.'
)
parser.add_argument('--print-countries',
                    dest='print_countries',
                    action='store_true',
                    help='Print available countries  in the  dataframe.')
parser.add_argument(
    '--countries',
    metavar='COUNTRIES',
    dest='countries',
    required=False,
    nargs='*',
    help=
    'Plot data for given countries. Pass arguments by name e.g. \'Austria\'. Multiple arguments can be passed in using whitespace character as delimeter, e.g. \'Austria Greece Sweden\''
)
parser.add_argument(
    '--continents',
    metavar='CONTINENTS',
    dest='continents',
    required=False,
    nargs='*',
    help=
    'Plot data for given countries. Pass arguments by name e.g. \'Austria\'. Multiple arguments can be passed in using whitespace character as delimeter, e.g. \'Austria Greece Sweden\''
)
parser.add_argument(
    '--x-axis',
    metavar='X_AXIS',
    dest='x_axis',
    required=False,
    default='date',
    help=
    'Define the x-axis; i.e. choose the data value to be defined as the plot\'s x-axis.'
)
parser.add_argument(
    '--y-axis',
    metavar='Y_AXIS',
    dest='y_axis',
    required=False,
    help=
    'Define the y-axis; i.e. choose the data value to be defined as the plot\'s y-axis.'
)

# parse cmd args
args = parser.parse_args()

# load the dataframe
df = load_dataframe(args.csv_file)
assert (all(df.columns == csv_header))

# print columns if needed
if args.print_columns:
    print('Data Columns /Statistics Available:')
    for idx, key in enumerate(df.columns):
        print('\t[{:3d}] -> {:}'.format(idx, key))
    sys.exit(0)

# print countries if needed
if args.print_countries:
    print('Countries Available in Dataset:')
    print_country_list(df)
    sys.exit(0)

## get the country list
country_list = args.countries if args.countries is not None else []

## get the continent list
continent_list = args.continents if args.continents is not None else []

if args.x_axis in geo_columns and args.y_axis in static_columns:
    create_bar(df,
               countries=country_list,
               continents=continent_list,
               columns=[args.x_axis, args.y_axis])
else:
    create_scatter(df,
                   countries=country_list,
                   continents=continent_list,
                   columns=[args.x_axis, args.y_axis])
