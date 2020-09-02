# Interactive Machine Learning Tool for Risk Factor Analysis

For Medical Data Analysis

---

## Dependencies

![Python](https://img.shields.io/badge/Python-^3.8-blue.svg?logo=python&longCache=true&logoColor=white&colorB=5e81ac&style=flat-square&colorA=4c566a)
![Flask](https://img.shields.io/badge/Flask-1.1.2-blue.svg?longCache=true&logo=flask&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![Flask-Assets](https://img.shields.io/badge/Flask--Assets-v2.0-blue.svg?longCache=true&logo=flask&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![Pandas](https://img.shields.io/badge/Pandas-v1.0.4-blue.svg?longCache=true&logo=python&longCache=true&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![Dash](https://img.shields.io/badge/Dash-v1.12.0-blue.svg?longCache=true&logo=python&longCache=true&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![Plotly](https://img.shields.io/badge/Plotly-v4.8.1-blue.svg?longCache=true&logo=python&longCache=true&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![GitHub Last Commit](https://img.shields.io/github/last-commit/google/skia.svg?style=flat-square&colorA=4c566a&colorB=a3be8c)

## Dataset
* [Behavioral Risk Factor Surveillance System (BRFSS 2018)](https://www.cdc.gov/brfss/annual_data/annual_2018.html)
    + [Data Overview](https://www.cdc.gov/brfss/annual_data/2018/pdf/overview-2018-508.pdf)
    + [Data Cookbook](https://www.cdc.gov/brfss/annual_data/2018/pdf/codebook18_llcp-v2-508.pdf)

## Installation

**Installation via `requirements.txt`**:

*for Mac/Linux development*
```shell 
$ git clone https://github.com/Interactive-Media-Lab-Data-Science-Team/Data-Preprocessing-Tool.git
$ cd Data-Preprocessing-Tool
$ python3 -m venv myenv
$ source myenv/bin/activate
$ pip3 install -r requirements.txt
$ export FLASK_ENV=development
$ flask run
```

*for Windows development*
```shell 
$ git clone https://github.com/Interactive-Media-Lab-Data-Science-Team/Data-Preprocessing-Tool.git
$ cd Data-Preprocessing-Tool
$ python3 -m venv myenv
$ myenv/Scripts/activate
$ pip3 install -r requirements.txt
$ set FLASK_ENV=development
$ flask run
```

**Installation via [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/)**:

```shell
$ git clone https://github.com/Interactive-Media-Lab-Data-Science-Team/Data-Preprocessing-Tool.git
$ cd Data-Preprocessing-Tool
$ pipenv shell
$ pipenv update
$ flask run
```


-----

