# FIFA Player Stats – Spider Web Comparator

Compare football players with radar charts and ML predictions (Value, Potential, Similar Players).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Or double-click `run.bat` on Windows.

## Data

Place `trending_football_players.xlsx` or `updated_trending_football_players.xlsx` in the project folder or in `Downloads/archive/`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub and click **New app**
4. Select your repo `Fifa-Player_Stats`, branch `master`, main file `app.py`
5. Click **Deploy**

The app will build and run. Ensure your Excel data files are in the repo or update `app.py` to load data from a URL.
