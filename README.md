## Introduction

`acab19` is a dead-simple data plotig/browsing tool for COVID-19 datasets 
distributed by [ourworldindata](https://ourworldindata.org/coronavirus).

## The dataset

The script uses the coronavirus dataset prepared and distributed by 
[ourworldindata](https://ourworldindata.org/coronavirus) [1](#roseretal).
A complete overview of the dataset can be found on 
[owid covid-19-data](https://github.com/owid/covid-19-data/tree/master/public/data) 
github page. [Here](https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-codebook.csv) 
is a quite extensive list of the parameters/data recorded in the dataset.

## The script

### Installation

The script is written in Python and uses the [pandas](https://pandas.pydata.org/) and 
[matplotlib](https://matplotlib.org/) libraries for analyzing and plotting the 
data.

Since this is a script it does not need installation; you need however to have 
all of its dependencies available. Best way to do that, is to install 
[pipreqs](https://pypi.org/project/pipreqs/) and use it to manage dependancies. 
That is in the root folder (aka the folder containing the script), run:
`pipreqs .`; this will create a `requirements.txt` file. Then run [pip](https://pypi.org/project/pip/) 
to install any missing modules: `pip install -r requirements.txt`. You should 
now be good to go!

## Bar graphs

Data columns/records which are not changing in time are plotted as bar graphs, 
where the x-axis is a list of selected countries and the y-axis the values of 
the selected data-record. These records are: `iso_code`,`continent`,`location`, 
`tests_units`,`population`,`population_density`,`median_age`,`aged_65_older`, 
`aged_70_older`,`gdp_per_capita`,`extreme_poverty`,`cardiovasc_death_rate`, 
`diabetes_prevalence`,`female_smokers`,`male_smokers`,`handwashing_facilities`, 
`hospital_beds_per_thousand`,`life_expectancy` and `human_development_index`.

E.g. to plot male smokers (%) for Greece, Spain and Italy, use:
`acab19.py --countries Greece Spain Italy --x-axis=location --y-axis=male_smokers` 
which will produce:
![alt text](https://github.com/xanthospap/acab19/blob/main/gallery/gr_sp_it_male_smokers.png?raw=true)

> When plotting a bar graph, the x-axis must be specified (via a command line argument) and set to `--x-axis=location`

## Scatter graphs

Data columns/records which are changing in time, are plotted as (line connected) 
scatter graphs. The x-axis in this case is the date column and the y-axis is the 
chosesn data record. For each selected country, we plot one line.

E.g. to plot the fully vaccinated in Greece, Spain, Italy and Israel, use:
`./acab19.py --countries Greece Spain Italy Israel --y-axis=people_fully_vaccinated` 
which will produce something like:
![alt text](https://github.com/xanthospap/acab19/blob/main/gallery/gr_sp_it_is_people_vaccinated.png?raw=true)

> When plotting a scatter graph, the x-axis can be ommited; if not, it must be set to `--x-axis=date`

## Dataset file

When triggered, the script will download the latest dataset file from [ourworldindata](https://ourworldindata.org/coronavirus) 
in csv format, aka [owid-covid-data.csv](https://covid.ourworldindata.org/data/owid-covid-data.csv). 
Users can skip the download (e.g. they have already performed a program run) via the 
`--csv-file=my_local_owid-covid-data.csv` switch.

The script has a couple of switches users can turn on to briefly browse elementary info:
  
  * to see the __full list of records/columns__ use the `--print-columns` switch, and
  * to see the __full list of countries/locations__ available, use `--print-countries`

## Special locations

_Special locations_ are considered the locations within the dataset that are marked 
with an ISO code starting with 'OWID_' (e.g. `OWID_AFR`). These may be listed 
via the `--print-countries` switch.

## References
[1] <a id="roseretal"></a> Max Roser, Hannah Ritchie, Esteban Ortiz-Ospina and Joe Hasell (2020) - "Coronavirus Pandemic (COVID-19)". Published online at OurWorldInData.org. Retrieved from: 'https://ourworldindata.org/coronavirus' [Online Resource]
