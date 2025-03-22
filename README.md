# mid-senior-data-scientist-interview

## How to run pipeline in VScode terminal

1. Create a virtual environment with Python 3.10 (since some libraries might have dependency issues with other versions):
```bash
python3.10 -m venv venv
```
2. Activate the virtual environment:
- On macOS/Linux
    ```bash
    source venv/bin/activate
    ```
3. Install dependecies using

```bash
pip3 install -r requirements.txt
```

## Setting Up the Environment Using Conda

Create the Conda environment using the provided suggested_env.yaml file:
```bash
conda env create -f suggested_env.yaml
```


## Run script from root, using
```bash
python3 -m src.run_pipeline
```

- This will create two files:
    1. One with preprocessed and engineered features.
    2. A sparse matrix containing the TF-IDF matrix from the company_description column, along with new numeric features.

