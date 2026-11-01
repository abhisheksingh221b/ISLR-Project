# UN Vote Forecaster – Streamlit dashboard

An interactive dashboard for exploring UN General Assembly voting blocs and vote predictions (1946–2019).

## Pages
- **Voting blocs** – world map and PCA map of the three voting blocs in each decade
- **Compare countries** – how often any two countries cast the same vote, year by year
- **Predict a resolution** – predicted vs actual votes for any 2010–2019 resolution, swing countries, and a SHAP explanation for any country
- **Model performance** – leaderboard, confusion matrix, and a live precision/recall slider for swing flags

## Folder layout
```
app.py
requirements.txt
app_data/        <- files created by the notebook (section 6)
```

## Run on your laptop
```
pip install -r requirements.txt
streamlit run app.py
```
It opens in your browser at http://localhost:8501.

## Put it online (free) with Streamlit Community Cloud
1. Create a new public GitHub repository and upload `app.py`, `requirements.txt` and the whole `app_data` folder.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick your repository, set the main file to `app.py`, and click **Deploy**.
4. After a few minutes you get a public link you can share.

## Updating the data
If you change the notebook, run it again in Colab. Section 6 creates a fresh `app_data` folder.
Download it (Files panel → right-click `app_data` → Download, or zip it first with `!zip -r app_data.zip app_data`)
and replace the old folder.

Data: UN General Assembly voting data compiled by Erik Voeten and colleagues, shared through TidyTuesday (March 2021).
