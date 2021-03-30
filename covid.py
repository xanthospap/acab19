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
import pandas
import argparse
import matplotlib.pyplot as plt

#https://covid.ourworldindata.org/data/owid-covid-data.csv
COVID_CSV_URI_E = 'https://opendata.ecdc.europa.eu/covid19/nationalcasedeath_eueea_daily_ei/csv/data.csv'
COVID_CSV_URI_G = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv/data.csv'

def population_col(dataset):
    """ Depending on the dataset (aka european/global) return the column
        name holding population data.
    """
    return 'popData2020' if dataset.lower() == 'european' else 'popData2019'

def data_source(dataset):
    """ Return the URI of the dource data (csv) given the dataset choice (aka
        european or global.
    """
    return COVID_CSV_URI_E if dataset.lower() == 'european' else COVID_CSV_URI_G

def get_csv(dataset, filename=None):
    """ Download the dataset in csv format; choose between the 'european' or 
        'global' dataset.
        If filename is provided, save the downloaded file as 'filename'.
    """
    if not filename:
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filename = 'covid_{}.csv'.format(timestamp_str)
    COVID_CSV_URI = COVID_CSV_URI_E if dataset.lower() == 'european' else COVID_CSV_URI_G
    print('Downloading dataset file {:} to {:} ...'.format(COVID_CSV_URI, filename), end='')
    with urllib.request.urlopen(COVID_CSV_URI) as response, open(
            filename, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print(' done!')
    return filename


def get_countries(pdf):
    """ Return unique countries in dataframe (via countriesAndTerritories 
        column).
    """
    return pdf.countriesAndTerritories.unique()

def add_mean(pdf, dataset, return_only_avg):
    """ Create a copy of the dataframe with new rows containing average values
        for deaths and cases.
        For each date in the dataframe create a new row (for this date) where
        the cases and deaths columns will be the average values for this date.
        The date/computation is skipped if less that 3/4 of the total number 
        of countries has data for this date.
        The value for the population column will be the average population of
        the countries contributing to the average computation.
        Values for the countriesAndTerritories and continentExp columns are 
        entered as Average and Avrg
        dataset is either european or global depending on the dataframe parsed.
        We return a new dataframe were the average values are included OR in
        case return_only_avg is True only return the newly computed values in
        a new dataframe (having the same column names as the original).
    """
    popData = population_col(dataset)
    tmp_df = {
        'dateRep': [],
        'cases': [],
        'deaths': [],
        'countriesAndTerritories': [],
        popData: [],
        'continentExp': []
    }
    dates_list = pdf.dateRep.unique()
    max_countries = pdf.countriesAndTerritories.unique().shape[0]
    for d in dates_list:
        countries_in_date = pdf.loc[pdf['dateRep'] == d].shape[0]
        if countries_in_date < 0.75 * max_countries:
            print(
                '[WARNING] Skipping mean for date {:} due to limited data, {:} out of {:}'
                .format(d, countries_in_date, max_countries))
        else:
            csdt = pdf.loc[pdf['dateRep'] == d,
                           ['cases', 'deaths', popData]].mean()
            for tpl in zip([
                    'dateRep', 'cases', 'deaths', 'countriesAndTerritories',
                    popData, 'continentExp'
            ], [
                    d,
                    int(csdt[0]),
                    int(csdt[1]), 'Average',
                    int(csdt[2]), 'Avrg'
            ]):
                tmp_df[tpl[0]].append(tpl[1])
    if return_only_avg:
        return pandas.DataFrame(tmp_df)
    return pdf.append(pandas.DataFrame(tmp_df), ignore_index=True)


def get_col_by_country(pdf, col, add_min_date, dataset, c_name='', c_code=''):
    if c_name == 'Average' or c_code == 'AVRG':
        pdf = add_mean(pdf, dataset, True)
    """
    df = pdf.loc[(pdf['countriesAndTerritories'] == c_name) |
                 (pdf['countryterritoryCode'] == c_code), ['dateRep', col]]
    """
    if col in ['deaths', 'cases']:
        tcol = col
        df = pdf.loc[(pdf['countriesAndTerritories'] == c_name), ['dateRep', col]]
    else:
        tcol = 'deaths'
        df = pdf.loc[(pdf['countriesAndTerritories'] == c_name), ['dateRep', tcol, population_col(dataset)]]
    if df.size == 0:
        print('[WARNING] No data for country: {:}/{:}'.format(c_name, c_code),
              file=sys.stderr)
        return None
    min_date = df['dateRep'].min()
    if add_min_date:
        ## add value of min date to every subsequent column (in-place)
        offset = df.loc[df['dateRep'].idxmin(), tcol]
        df.loc[df['dateRep'] > min_date, tcol] += offset
    else:
        ## remove min date row (in-place)
        df.drop(df[df['dateRep'] == min_date].index, inplace=True)
    if col == 'deathsPerMillion' or col == 'dppm':
        df['dppm'] = df['deaths'] * 1e6 / df[population_col(dataset)]
    return df


def country_chart(country_list, dataset, add_min_date, col, pdf):
    plt.figure()
    legend = []
    for cntr in country_list:
        df = get_col_by_country(pdf, col, add_min_date, dataset, cntr, cntr)
        if df is not None:
            legend.append(cntr)
            plt.plot(df['dateRep'], df[col], marker='o')
    plt.legend(legend)
    plt.title('Coivd-19 {:}; data source:\n{:}'.format(col, data_source(dataset)))
    plt.grid(True)
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
                    help='Use this data csv file instead of dwonloading.')

parser.add_argument(
    '--countries',
    metavar='COUNTRIES',
    dest='countries',
    required=False,
    nargs='*',
    help=
    'Plot data for given countries. Pass arguments eithet by name e.g. \'Austria\' or by code e.g. \'AUT\'. Multiple arguments can be passed in using whitespace character as delimeter, e.g. \'Austria GR Sweden\'. To also plot the European average, use the \'Average\' keyword (e.g. \'Austria GR Sweden Average\'.'
)

parser.add_argument(
    '--cumulative',
    dest='cumulative',
    action='store_true',
    help=
    'Plot cumulative values, aka add new daily values to the ones reported in the minimum date. If not set, then the script will only plot new daily values.'
)

parser.add_argument('--plot-value',
                    metavar='PLOT_VALUE',
                    dest='plot_value',
                    required=False,
                    choices=['deaths', 'cases', 'dppm'],
                    default='cases',
                    help='Choose wich data value to plot.')
parser.add_argument('--dataset',
                    metavar='DATASET',
                    dest='dataset',
                    required=False,
                    choices=['european', 'global'],
                    default='european',
                    help='Choose wich data value to plot.')

# parse cmd args
args = parser.parse_args()

## download csv data and get the local filename (if needed)
csvf = args.csv_file if args.csv_file is not None else get_csv(args.dataset)

## get the country list
country_list = args.countries

## load csv via pandas
#dateRep,day,month,year,cases,deaths,countriesAndTerritories,geoId,countryterritoryCode,popData2020,continentExp
usecols = [
    'dateRep', 'cases', 'deaths', 'countriesAndTerritories',
    'popData2020', 'continentExp'
]
if args.dataset == 'global':
    usecols[usecols.index('popData2020')] = 'popData2019'
df = pandas.read_csv(
    csvf,
    usecols=usecols,
    parse_dates=[0],
    date_parser=lambda d: datetime.datetime.strptime(d, '%d/%m/%Y'))

country_chart(country_list, args.dataset, args.cumulative, args.plot_value, df)
