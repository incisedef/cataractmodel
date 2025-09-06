# Cataract Classification Project

This project classifies cataract eye images into **Mature** and **Immature** stages using CNN.

## Structure
- `cataract_dataset/` → Dataset folder (kaggle) from cataract dataset path:
path = kagglehub.dataset_download("akshayramakrishnan28/cataract-classification-dataset")

## Classification Model
- `prepare_dataset/` → classifying as "immature" and "mature"
- `train/` → Model training & preprocessing scripts
- `.venv/` → Virtual environment

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
