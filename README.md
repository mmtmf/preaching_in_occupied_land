# Preaching in Occupied Land  
**Topic Modelling & Sentiment Analysis of Danish Sermons During WWII**

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

> A digital humanities project that investigates emotional and thematic trends in Danish sermons delivered during the Second World War.

---

## Table of Contents
- [What This Project Does](#what-this-project-does)
- [Data](#data)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Outputs](#outputs)
- [Dependencies](#dependencies)
- [Citation & License](#citation--license)
- [About the Project](#about-the-project)

---

## What This Project Does

This project analyzes historical sermon texts through two main pipelines:

### Sentiment Analysis (`sentiment.py`)
- Applies **[Sentida](https://github.com/Guscode/Sentida)**, a Danish sentiment analyzer.
- Smooths sentiment trends using LOWESS.
- Evaluates statistical significance with t-tests and SEM.
- Detects **change points** using `ruptures`.
- Generates **annotated visualizations** highlighting key WWII events and affiliations.

### Topic Modeling Analysis (`topics_over_time.py`)
- Uses preprocessed LDA results from **[DARIAH TopicsExplorer](https://dariah-de.github.io/TopicsExplorer/)**.
- Tracks topic distributions over time across sermons.
- Compares topic shifts before and after major historical events (e.g., government collapse, censorship).
- Produces CSVs and **event-aligned topic trend plots**.

---

## Data

Located in the `data/` folder:

Due to copyright regulations, the actual sermon texts and metadata are **not stored** in this public repository.  
Contact the author for relevant insight into these files.

In this repository you’ll find:
- Overview of sermons and relevant references: `Sermon and references.pdf`
- Topic model outputs: `topics_csv.CSV`, `topics.csv`, `topics-similarities_csv.CSV`, `document_topic-distribution_csv.CSV`, `document_similarities_csv.CSV`
- Stopwords: `stopwords.txt`

---

## Getting Started

Clone the repo and install dependencies in a virtual environment:

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

## Usage

Folder structure
```
- data
    - præ_fin
        - pr43_17.txt
        ...
    - stopwords.txt
    - metadata_v40.csv
    - topics_csv.CSV
    - topics.csv
    - topics-similarities_csv.CSV
    - document_topic-distribution_csv.CSV 
    - document_similarities_csv.CSV
- results
    - results_sentiment
        - ...
    - results_topics
        - ...
- src
    - sentiment.py
    - topics_over_time.py
```

Install dependencies:
```bash
pip install -r requirements.txt
```

▶️ Run scripts:
 
```bash
python src/sentiment.py
python src/topics_over_time.py
```

## Outputs

Saved in the `results/` directory:

- Sentiment trend graphs (by affiliation, event)
- Change point detection visualizations
- Pre/Post event topic comparison plots
- CSVs with **FDR-corrected t-tests** and most changed topics
- Etc.

---

## Dependencies
See requirements.txt

## Citation & License
**License:**  
This project is licensed under the [MIT License](LICENSE.txt).

**Citation:**  
If you use this project in academic work, please cite it as:

> Thunbo, Michael Mørch. *Preaching  in Occupied Land: Crisis and Sermons in Wartime Denmark (1940–1945)*. GitHub repository, 2025. https://github.com/mmtmf/preaching-in-occupied-land

## About the Project

Developed as part of a PhD research project examining sermons in times of national crisis.

