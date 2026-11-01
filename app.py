# UN Vote Forecaster - Streamlit dashboard
# Run with:  streamlit run app.py

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title='UN Vote Forecaster', page_icon='🗳️', layout='wide')

# The data files can sit next to app.py or inside an "app_data" folder
FOLDER = Path(__file__).parent
DATA = FOLDER / 'app_data' if (FOLDER / 'app_data').exists() else FOLDER

VOTE_COLORS = {'yes': 'green', 'abstain': 'orange', 'no': 'red'}

# Readable names for the model's inputs (used in the "why" chart)
NICE_NAMES = {
    'year': 'Year',
    'importantvote': 'Important vote',
    'topic_co': 'Topic: colonialism',
    'topic_di': 'Topic: arms control',
    'topic_ec': 'Topic: economic development',
    'topic_hr': 'Topic: human rights',
    'topic_me': 'Topic: Palestinian conflict',
    'topic_nu': 'Topic: nuclear weapons',
    'past_yes_rate': '"Yes" share, last 3 sessions',
    'past_no_rate': '"No" share, last 3 sessions',
    'past_abstain_rate': 'Abstain share, last 3 sessions',
    'topic_yes_rate': '"Yes" share on this topic',
    'topic_no_rate': '"No" share on this topic',
    'topic_abstain_rate': 'Abstain share on this topic',
    'agree_us': 'Agreement with the US',
    'agree_russia': 'Agreement with Russia',
    'agree_china': 'Agreement with China',
    'agree_india': 'Agreement with India',
    'topic_world_yes_rate': 'World "yes" share on this topic',
    'similarity': 'Similarity to closest past resolution',
    'similar_my_vote': 'Its vote on the most similar resolution',
    'similar3_my_vote': 'Its average vote on the 3 most similar',
    'similar_world_yes': 'World "yes" share on the similar resolution',
    'similar_us_vote': 'US vote on the similar resolution',
}


# ---------------------------------------------------------------
# Load everything once (Streamlit remembers it between clicks)
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    blocs = pd.read_csv(DATA / 'blocs.csv')
    votes = pd.read_parquet(DATA / 'votes.parquet')
    preds = pd.read_parquet(DATA / 'predictions.parquet')
    resolutions = pd.read_csv(DATA / 'resolutions.csv')
    leaderboard = pd.read_csv(DATA / 'leaderboard.csv')
    confusion = pd.read_csv(DATA / 'confusion.csv', index_col=0)
    with open(DATA / 'features.json') as f:
        features = json.load(f)
    return blocs, votes, preds, resolutions, leaderboard, confusion, features


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(DATA / 'xgb_model.json')
    explainer = shap.TreeExplainer(model)
    return explainer


blocs, votes, preds, resolutions, leaderboard, confusion, features = load_data()


# ---------------------------------------------------------------
# Page 1: Voting blocs
# ---------------------------------------------------------------
def blocs_page():
    st.title('Voting blocs at the UN')
    st.write('Countries grouped by how they vote (PCA + K-Means, 3 groups per decade).')

    decade = st.select_slider('Decade', options=sorted(blocs['decade'].unique()), value=1980)
    d = blocs[blocs['decade'] == decade]

    world_map = px.choropleth(d, locations='iso3', color='bloc', hover_name='country',
                              title=f'Voting blocs in the {decade}s')
    st.plotly_chart(world_map)

    pca_map = px.scatter(d, x='x', y='y', color='bloc', hover_name='country',
                         title='PCA map: countries close together vote alike')
    st.plotly_chart(pca_map)

    st.subheader('Who is in each bloc?')
    for bloc_name in d['bloc'].unique():
        members = sorted(d[d['bloc'] == bloc_name]['country'])
        with st.expander(f'{bloc_name} ({len(members)} countries)'):
            st.write(', '.join(members))


# ---------------------------------------------------------------
# Page 2: Compare countries
# ---------------------------------------------------------------
def agreement_by_year(country_a, country_b):
    """Percent of votes where the two countries voted the same way, for each year."""
    a = votes[votes['country'] == country_a]
    b = votes[votes['country'] == country_b]
    both = a.merge(b, on=['rcid', 'year'], suffixes=('_a', '_b'))
    both['same'] = both['vote_a'] == both['vote_b']
    return both.groupby('year')['same'].mean() * 100


def compare_page():
    st.title('How often do countries vote the same way?')
    countries = sorted(votes['country'].unique())
    main = st.selectbox('Country', countries, index=countries.index('India'))
    others = st.multiselect('Compare with', countries, default=['United States', 'Russia', 'China'])

    if not others:
        st.info('Pick at least one country to compare with.')
        return

    rows = []
    for other in others:
        yearly = agreement_by_year(main, other)
        for year, value in yearly.items():
            rows.append({'year': year, 'with': other, 'agreement': value})
    table = pd.DataFrame(rows)

    # 3-year average makes the lines smoother
    table['smoothed'] = table.groupby('with')['agreement'].transform(lambda s: s.rolling(3, min_periods=1).mean())

    chart = px.line(table, x='year', y='smoothed', color='with',
                    labels={'smoothed': '% same vote (3-year average)'},
                    title=f'{main}: share of votes cast the same way')
    st.plotly_chart(chart)

    st.subheader('Average by decade (%)')
    table['decade'] = table['year'] // 10 * 10
    st.dataframe(table.pivot_table(index='with', columns='decade', values='agreement').round(0))


# ---------------------------------------------------------------
# Page 3: Predict a resolution
# ---------------------------------------------------------------
def predict_page():
    st.title('Predict how countries vote on a resolution')
    st.write('The model learned from 1980-2009 and never saw these 2010-2019 votes.')

    year = st.selectbox('Year', sorted(preds['year'].unique(), reverse=True))
    ids_this_year = preds[preds['year'] == year]['rcid'].unique()
    options = resolutions[resolutions['rcid'].isin(ids_this_year)]
    titles = dict(zip(options['rcid'], options['descr'].fillna(options['short']).str[:110]))
    rcid = st.selectbox('Resolution', list(titles), format_func=lambda r: titles[r])

    info = resolutions[resolutions['rcid'] == rcid].iloc[0]
    st.write('**Resolution:**', info['descr'])
    st.caption(f"Similarity to the closest earlier resolution: {info['similarity']:.2f}")

    r = preds[preds['rcid'] == rcid]
    col1, col2, col3 = st.columns(3)
    col1.metric('Countries voting', len(r))
    col2.metric('Correct predictions', f"{(r['vote'] == r['predicted']).mean():.0%}")
    col3.metric('Swing countries flagged', int(r['swing_flag'].sum()))

    which = st.radio('Show on the map', ['predicted', 'vote'], horizontal=True,
                     format_func=lambda c: 'Predicted vote' if c == 'predicted' else 'Actual vote')
    vote_map = px.choropleth(r, locations='iso3', color=which, hover_name='country',
                             color_discrete_map=VOTE_COLORS)
    st.plotly_chart(vote_map)

    st.subheader('Swing countries: most likely to change their usual vote')
    swing = r.sort_values('p_change', ascending=False).head(10)
    swing = swing[['country', 'vote', 'predicted', 'p_change', 'swing_flag']].copy()
    swing['p_change'] = (swing['p_change'] * 100).round(0)
    swing.columns = ['Country', 'Actual vote', 'Predicted vote', 'Chance of change (%)', 'Flagged']
    st.dataframe(swing, hide_index=True)

    explain_one_country(r)


def explain_one_country(r):
    st.subheader('Why did the model predict this?')
    countries = sorted(r['country'])
    who = st.selectbox('Pick a country', countries,
                       index=countries.index('India') if 'India' in countries else 0)
    row = r[r['country'] == who].iloc[0]
    st.write(f"**{who}** actually voted **{row['vote']}**. The model predicted **{row['predicted']}**.")

    # SHAP: how much each input pushed the prediction towards the predicted vote
    explainer = load_model()
    inputs = row[features].to_frame().T.astype(float)
    class_number = ['no', 'abstain', 'yes'].index(row['predicted'])
    shap_values = explainer(inputs).values[0, :, class_number]

    why = pd.DataFrame({'input': [NICE_NAMES[f] for f in features], 'effect': shap_values})
    why = why.reindex(why['effect'].abs().sort_values().index).tail(8)   # 8 biggest effects
    why['direction'] = ['towards' if e > 0 else 'away from' for e in why['effect']]

    bar_chart = px.bar(why, x='effect', y='input', color='direction', orientation='h',
                       title=f'What pushed the prediction towards or away from "{row["predicted"]}"')
    st.plotly_chart(bar_chart)


# ---------------------------------------------------------------
# Page 4: Model performance
# ---------------------------------------------------------------
def performance_page():
    st.title('How good are the predictions?')
    st.write('Tested on 2010-2019 votes. About 80% of votes are "yes", so macro F1 is the main score.')

    st.subheader('Leaderboard')
    st.dataframe(leaderboard, hide_index=True)

    st.subheader('XGBoost confusion matrix (rows = actual, columns = predicted)')
    st.dataframe(confusion)

    st.subheader('Swing flags: precision vs recall')
    threshold = st.slider('Flag a country when its chance of change is above', 0.1, 0.9, 0.5, 0.05)
    known = preds[preds['similar_my_vote'].notna()]          # votes where we know the old vote
    flagged = known[known['p_change'] > threshold]
    changed = known[known['changed']]

    precision = flagged['changed'].mean() if len(flagged) > 0 else 0
    recall = (changed['p_change'] > threshold).mean()

    col1, col2, col3 = st.columns(3)
    col1.metric('Countries flagged', f'{len(flagged):,}')
    col2.metric('Precision', f'{precision:.0%}', help='Of the flagged countries, how many really changed')
    col3.metric('Recall', f'{recall:.0%}', help='Of the countries that changed, how many were flagged')


# ---------------------------------------------------------------
# Sidebar menu
# ---------------------------------------------------------------
st.sidebar.title('🗳️ UN Vote Forecaster')
page = st.sidebar.radio('Go to', ['Voting blocs', 'Compare countries', 'Predict a resolution', 'Model performance'])
st.sidebar.caption('Data: UN General Assembly votes, 1946-2019 (Voeten et al., via TidyTuesday).')

if page == 'Voting blocs':
    blocs_page()
elif page == 'Compare countries':
    compare_page()
elif page == 'Predict a resolution':
    predict_page()
else:
    performance_page()
