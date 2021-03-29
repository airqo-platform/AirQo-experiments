# Reproducible Research Template

This folder contains a basic template file and folder structure for reproducible resreach in Python. It consists of the following folders and files:

## `code/`

**Mandatory** 

- Must contain Jupyer notebooks and / or Python scripts.
- There must (at a minimum) be clear seperation of function between data download, data munging and analysis. In the case of Jupyter notebooks these may be seperate headings, for scripts they may be seperate files.
- ToDo: Structure for tests and Python modules.

## `data/`

### `raw/`

**Mandatory**

- Must contain eiter the raw data used for this research, or a reference to that data if it cannot be uploaded (e.g. it's too big or it's confidential).

### `clean/`

- If the analysis is not performed directly on raw data, this folder should contain "clean" (munged, combined) data.

## LICENSE

- If this folder is not in a repository which already has a license, an appropriate license is essential.

## `readme.md`

- Instructions to run the code, for example:

> Clone this repository:
> ```
> git clone https://github.com/airqo-platform/AirQo-experiments.git
> ```
> Change directory to this folder:
> ```
> cd reproducibility-template
> ```
> Create and activate clean conda enviroment:
> ```
> conda create --force -n reproducibility-template python=3.6
> conda activate reproducibility-template
> ```
> Install requirements:
> ```
> pip install -r requirements.txt
> ```
> Execute code:
> ```
> python code/01_get_data.py
> python code/02_clean_data.py
> python code/03_analysis.py
> ```


Substitute `environment.yml` for `requirements.txt`, if appropriate. If using Jupyter notebooks, instructions may be better embedded within the notebook file.

## `requirements.txt`