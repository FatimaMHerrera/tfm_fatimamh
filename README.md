
# TFM Environment Setup Guide

---
__Author__: Fatima Concepcion Mesa Herrera
---

This document provides detailed instructions on how to set up and use the environment for the TFM (Final Master's Project).

[Github repository tfm_fatimamh](https://github.com/FatimaMHerrera/tfm_fatimamh.git)

## File System

This project in development phase so it is not yet finished.
For this reason all this project need to be in a folder called `TFM_FATIMA`.


The project's file system is organized as follows:

- `data/`: Contains the input and output data of the project.
    - `inputs/`: Contains the input data for the project. Suggested inputs by the Kaggle competition.
    
    - `outputs/`: Contains the output data of the project. Here, trained models and the best parameters for each of them after optimizing with Optuna will be saved.
        - `best_models_joblibs`: Contains the best models after optimizing with Optuna.
        - `best_parameters_jsons`: Contains the best parameters for each of the models after optimizing with Optuna.
        - `metrics`: Contains the metrics of the models after optimizing with Optuna.
        - `pruebas`: Contains the backup.
        - `graphics`: Contains the graphics of the TFM.
        - `study_optuna`: Contains the studies of the optimization with Optuna.

- `src/`: Contains the source code of the project.
    - `full_results.ipynb`: Contains the main Notebook of the project.
    - `api_web`: Contains the code for the web api.
      - `app.py`: Contains the code for the web api.

- `README.md`: This file.  
- `requirements.txt`: Contains the dependencies of the project.


## Python based

This project is based on Python 3.11.5. It is recommended to use the latest version of Python 3.11.

## Using venv (Python Virtual Environment)

### Creating a virtual environment with venv

#### On Windows:

```bash
python -m venv TFM_ENVIRONMENT
```

#### On Linux:

```bash
python3 -m venv TFM_ENVIRONMENT
```

### Activating the virtual environment

#### On Windows:

```bash
TFM_ENVIRONMENT\Scripts\activate
```

#### On Linux:

```bash
source TFM_ENVIRONMENT/bin/activate
```

## Using Conda

### Creating a virtual environment with Conda

```bash
conda create --name TFM_ENVIRONMENT python=3.11
```

### Activating the virtual environment with Conda

#### On Windows and Linux:

```bash
conda activate TFM_ENVIRONMENT
```

## Installing Dependencies

Once the virtual environment is activated, install the necessary dependencies by running the following command at the root of the project, where the `requirements.txt` file is located:

```bash
pip install -r requirements.txt
```

## Deactivating the Virtual Environment

When you have finished working on the project, you can deactivate the virtual environment with the following commands:

#### For venv:

#### On Windows:
```bash
deactivate
```

#### On Linux:
```bash
deactivate
```

#### For Conda:

#### On Windows and Linux:
```bash
conda deactivate
```
---

Also, you can run a Docker build.

- Navigate to the path where the Dockerfile is located, then execute the command `docker build -t fatima_tfm_img .`.

- Wait for the build process to complete, and then run the following command: `docker run -p 8888:8888 fatima_img`


---
This `README.md` provides a basic guide and may need to be adjusted or expanded according to the specific needs of your project. Don't forget to check the specific paths and commands for your operating system and configuration.


## Running the project.
The normal workflow would be to create the environment, install the dependencies, and gradually run `full_results.ipynb`, changing some parameters to see the reproducibility of the work.
