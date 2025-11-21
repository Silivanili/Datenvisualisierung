# \Datenvisualisierung\run.py
from app import app as dash_app
from app.layout import app_layout

dash_app.layout = app_layout()

import app.callbacks  

if __name__ == "__main__":
    dash_app.run()
