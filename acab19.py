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
import operator
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


def set_pandas_display_options():
    """Set pandas display options."""
    # Ref: https://stackoverflow.com/a/52432757/
    display = pandas.options.display

    display.max_columns = 1000
    display.max_rows = 1000
    display.max_colwidth = 199
    display.width = 1000
    # display.precision = 2  # set as needed


def str_to_operator(op_str):
    if op_str == '=':
        return operator.eq
    elif op_str == '<':
        return operator.lt
    elif op_str == '<=':
        return operator.le
    elif op_str == '>=':
        return operator.ge
    elif op_str == '>':
        return operator.gt
    elif op_str == 'or':
        return operator.or_
    elif op_str == 'and':
        return operator.and_
    err_msg = '[ERROR] Invalid comparisson operator \'{:}\''.format(op_str)
    raise RuntimeError(err_msg)


def apply_filter(df, filter_dct):
    if 'target' in filter_dct:  ## simple filter
        return df[filter_dct['op'](df[filter_dct['target']], filter_dct['val'])]
    else:  ## filter has sub-filters
        sub_filters = max([
            int(re.match('filter([0-9]+)', kkey).group(1))
            for kkey in filter_dct
            if re.match('filter[0-9]+', kkey)
        ]) + 1  ## filters a 0-indexed
        if sub_filters > 2:
            raise RuntimeError(
                '[ERROR] Parenthesized filter criteria must be at maximum two!')
        op = str_to_operator(filter_dct['con0-1'])
        """
        print('df[{:}('.format(op), end='')
        print('({:}(df[{:}], {:})),'.format(filter_dct['filter0']['op'], filter_dct['filter0']['target'], filter_dct['filter0']['val']), end='')
        print('({:}(df[{:}], {:})))]'.format(filter_dct['filter1']['op'], filter_dct['filter1']['target'], filter_dct['filter1']['val']))
        """
        return df[op(
            (filter_dct['filter0']['op'](df[filter_dct['filter0']['target']],
                                         filter_dct['filter0']['val'])),
            (filter_dct['filter1']['op'](df[filter_dct['filter1']['target']],
                                         filter_dct['filter1']['val'])))]


def filter_countries(df, filter_str):
    num_filters, filter_dct = parse_user_filter(filter_str)
    ## first filter
    filtered_df = apply_filter(df, filter_dct['filter0'])
    for filter_nr in range(1, num_filters):
        key = 'filter{:}'.format(filter_nr)
        k_filter = filter_dct[key]
        new_df = apply_filter(df, k_filter)
        operation = filter_dct['con{:}-{:}'.format(filter_nr - 1, filter_nr)]
        if operation == 'or':
            # filtered_df = pandas.concat([filtered_df, new_df]).drop_duplicates()
            filtered_df = pandas.merge(filtered_df, new_df, how='outer')
        elif operation == 'and':
            filtered_df = pandas.merge(filtered_df, new_df, how='inner')
    # return filtered_df
    return list(
        flatten(filtered_df['location'].drop_duplicates(keep='last').values))


def parse_user_filter(filter_str):
    """ Example:
         filter_str = '(gdp_per_capita > 5 and gdp_per_capita < 7) and human_development_index > 1 or (cardiovasc_death_rate>5 and extreme_poverty<20)'
         Returned Value:
        {
            'filter0': 
                {
                'filter0': 
                    {
                        'target': 'gdp_per_capita', 'op': <built-in function gt>, 'val': 5.0 
                     }, 
                 'con0-1': 'and', 
                 filter1': 
                    {
                        'target': 'gdp_per_capita', 'op': <built-in function lt>, 'val': 7.0
                    }
                },
             'con0-1': 'and', 
             'filter1': 
                {
                    'target': 'human_development_index', 'op': <built-in function gt>, 'val': 1.0
                }, 
             'con1-2': 'or', 
             'filter2': 
                {'
                filter0': 
                    {
                        'target': 'cardiovasc_death_rate', 'op': <built-in function gt>, 'val': 5.0
                    }, 
                'con0-1': 'and', 
                'filter1': 
                    {
                        'target': 'extreme_poverty', 'op': <built-in function lt>, 'val': 20.0
                    }
                }
            }
    """
    dct = {}
    ## add spaces before and after symbols >, <, =, (, )
    filter_str = re.sub('(\()([a-z]+)', '\\1 \\2', filter_str)
    filter_str = re.sub('([0-9]+)(\))', '\\1 \\2', filter_str)
    for w in ['>', '<', '=']:
        filter_str = re.sub('({:})([a-z0-9]+)'.format(w), '\\1 \\2', filter_str)
        filter_str = re.sub('([a-z0-9]+)({:})'.format(w), '\\1 \\2', filter_str)
    fcols = filter_str.split()
    assert (len(fcols) > 0)
    idx = 0
    num_filters, sub_filter = 0, 0
    open_parenthesis = False
    while idx < len(fcols):
        if fcols[idx] == '(':
            assert (not open_parenthesis)
            open_parenthesis = True
            idx += 1
        d_column = fcols[idx]
        assert (d_column in static_columns)
        d_op = fcols[idx + 1]
        assert (d_op in ['=', '>', '<', '>=', '<='])
        d_val = float(fcols[idx + 2])
        idx += 3
        key = 'filter{:}'.format(num_filters)
        if open_parenthesis:
            if key not in dct:
                dct[key] = {}
            key2 = 'filter{:}'.format(sub_filter)
            dct[key][key2] = {
                'target': d_column,
                'op': str_to_operator(d_op),
                'val': d_val
            }
            sub_filter += 1
        else:
            dct[key] = {
                'target': d_column,
                'op': str_to_operator(d_op),
                'val': d_val
            }
            num_filters += 1
        if idx < len(fcols):
            if fcols[idx] == ')':
                assert (open_parenthesis)
                sub_filter = 0
                open_parenthesis = False
                num_filters += 1
                idx += 1
        if idx < len(fcols):
            con = fcols[idx]
            assert (con in ['and', 'or'])
            if open_parenthesis:
                dct[key]['con{:}-{:}'.format(sub_filter - 1, sub_filter)] = con
            else:
                assert (not open_parenthesis)
                dct['con{:}-{:}'.format(num_filters - 1, num_filters)] = con
            idx += 1
    return num_filters, dct


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
    assert (y_axis in csv_header and y_axis not in static_columns)
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
    plt.xticks(rotation=45, ha='right')
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
    'Plot data for given countries. Pass arguments by name e.g. \'Austria\'. Multiple arguments can be passed in using whitespace character as delimeter, e.g. \'Austria Greece Sweden\'. If left unset and \'--continents\' is also unset, then all countries in the data file are condidered.'
)
parser.add_argument(
    '--continents',
    metavar='CONTINENTS',
    dest='continents',
    required=False,
    nargs='*',
    help=
    'Plot data for given continents. Pass arguments by name e.g. \'Austria\'. Multiple arguments can be passed in using whitespace character as delimeter, e.g. \'Asia Africa\''
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
parser.add_argument('--filter',
                    metavar='FILTER',
                    dest='filter',
                    required=False,
                    nargs='*',
                    help='Filter locations based on some static column.')

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

## choose all countries if user-defined continets/countries lists are empty
if continent_list == [] and country_list == []:
    country_list = get_country_list(df)

## if we are going to filter the data, fuck the continents; collect all
## countries to the country list
if args.filter is not None:
    [
        country_list.extend(countries_in_continent(df, continent))
        for continent in continent_list
    ]
    continent_list = []
    ## apply filters to get (valid) countries
    filtered_country_list = filter_countries(df, *args.filter)
    ## get the intesection
    country_list = [
        country for country in country_list if country in filtered_country_list
    ]
    set_pandas_display_options()
    print(df.loc[df['location'].isin(country_list),
                 static_columns].drop_duplicates(subset=['location']))

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
