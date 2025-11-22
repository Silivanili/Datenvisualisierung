# Steam Dashboard  

A small Dash app that lets you explore a **Steam games** dataset with a few interactive pages.  

---

## Installation

```bash

git clone https://github.com/silivanili/Datenvisualisierung.git

cd steam-dashboard

pip install -r requirements.txt

python run.py
```

Open your browser and go to:

```
http://127.0.0.1:8050/
```

You should see the navigation bar at the top and the sidebar on the left.

---

## Download the data 

This project can not include the datasets directly. Please download all four CSV files from Kaggle:

**Dataset source:**
[https://www.kaggle.com/datasets/artermiloff/steam-games-dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)

After downloading, place all CSV files into:

```
app/data/
```

The app expects the files to be in this folder in order to load and display them.

---

## How to use it

1. **Select a dataset**  
   In the sidebar choose one of the CSV files listed under *Select Dataset* (for example `games_march2025_cleaned.csv`).  
   Click **Load Dataset**. The app will read the file.


All plots are automatically cached when possible, so switching pages or re‑applying filters is fast.

---

## Project structure

```
Datenvisualisierung/
│
├─ app/
│   ├─ __init__.py          # creates Dash app & Flask cache
│   ├─ cache.py             # filesystem cache configuration
│   ├─ config.py            # paths, constants
│   ├─ layout.py            # navbar, sidebar, page layouts
│   ├─ callbacks.py         # all Dash callbacks
│   ├─ plots.py             # helper functions for figures
│   ├─ utils.py             # empty_fig, df helpers
│   └─ data/
│       └─ processing.py    # data loading, parsing, cached aggregations
│
├─ run.py                    # entry‑point 
├─ requirements.txt
└─ README.md                
```
