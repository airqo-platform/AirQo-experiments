# Example of using calibration module in a notebook

## Installation

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (it also works with [anaconda](https://docs.anaconda.com/anaconda/install/), but we do not need the extra packages).

Clone this repository:

```
git clone https://github.com/airqo-platform/AirQo-experiments.git
```

Change to the calibration_notebooks folder:

```
cd AirQo-experiments/calibration_notebooks
```

With conda installed, run the following commands to create the virtual environment and activate it:

```
conda env create -f environment.yml
conda activate calibration_notebooks
```

## Running the notebook

Run:

```
jupyter-lab
```

Navigate to `import-example.ipynb` and run all. The code:

```
import calibration
```

should run without error.
