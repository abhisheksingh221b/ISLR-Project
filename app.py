import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title='UN Vote Forecaster', page_icon='🗳️', layout='wide')

# DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app_data') + os.sep 
DATA = os.path.dirname(os.path.abspath(__file__)) + os.sep   # data files sit next to app.py# works from any folder
VOTE_COLORS = {'yes': '#2e7d32', 'abstain': '#f9a825', 'no': '#c62828'}
TOPIC_NAMES = {'co': 'Colonialism', 'di': 'Arms control and disarmament', 'ec': 'Economic development',
               'hr': 'Human rights', 'me': 'Palestinian conflict', 'nu': 'Nuclear weapons', 'other': 'No topic label'}

# Plain-English names for the model's features
NICE_NAMES = {
    'year': 'Year',
    'importantvote': 'Marked as an important vote',
    'topic_co': 'Topic: colonialism',
    'topic_di': 'Topic: arms control',
    'topic_ec': 'Topic: economic development',
    'topic_hr': 'Topic: human rights',
    'topic_me': 'Topic: Palestinian conflict',
    'topic_nu': 'Topic: nuclear weapons',
    'past_yes_rate': 'Share of "yes" votes, last 3 sessions',
    'past_no_rate': 'Share of "no" votes, last 3 sessions',
    'past_abstain_rate': 'Share of abstentions, last 3 sessions',
    'topic_yes_rate': 'Past "yes" share on this topic',
    'topic_no_rate': 'Past "no" share on this topic',
    'topic_abstain_rate': 'Past abstain share on this topic',
    'agree_us': 'Recent agreement with the US',
    'agree_russia': 'Recent agreement with Russia',
    'agree_china': 'Recent agreement with China',
    'agree_india': 'Recent agreement with India',
    'topic_world_yes_rate': 'How much the world says "yes" on this topic',
    'similarity': 'How similar the closest past resolution is',
    'similar_my_vote': 'Its vote on the most similar past resolution',
    'similar3_my_vote': 'Its average vote on the 3 most similar',
    'similar_world_yes': 'World "yes" share on the similar resolution',
    'similar_us_vote': 'US vote on the similar resolution',
}


# ---------- load data (cached so it only happens once) ----------
@st.cache_data
def load_data():
    blocs = pd.read_csv(DATA + 'blocs.csv')
    votes = pd.read_parquet(DATA + 'votes.parquet')
    preds = pd.read_parquet(DATA + 'predictions.parquet')
    res = pd.read_csv(DATA + 'resolutions.csv')
    leaderboard = pd.read_csv(DATA + 'leaderboard.csv')
    confusion = pd.read_csv(DATA + 'confusion.csv', index_col=0)
    features = json.load(open(DATA + 'features.json'))
    return blocs, votes, preds, res, leaderboard, confusion, features


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(DATA + 'xgb_model.json')
    explainer = shap.TreeExplainer(model)
    return model, explainer


blocs, votes, preds, res, leaderboard, confusion, features = load_data()

st.sidebar.title('🗳️ UN Vote Forecaster')
page = st.sidebar.radio('Go to', ['Voting blocs', 'Compare countries', 'Predict a resolution', 'Model performance'])
st.sidebar.caption('Data: UN General Assembly votes, 1946–2019 (Voeten et al., via TidyTuesday).')


# =================================================================
if page == 'Voting blocs':
    st.title('Voting blocs at the UN')
    st.write('Countries grouped by how they vote, using PCA and K-Means (3 groups per decade). '
             'Each bloc is named after well-known countries inside it.')

    decade = st.select_slider('Decade', options=sorted(blocs['decade'].unique()), value=1980,
                              format_func=lambda d: f'{d}s')
    d = blocs[blocs['decade'] == decade]

    fig = px.choropleth(d.dropna(subset=['iso3']), locations='iso3', color='bloc', hover_name='country',
                        title=f'Voting blocs in the {decade}s')
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=40, b=0), legend_title_text='Bloc')
    st.plotly_chart(fig, width='stretch')

    left, right = st.columns([3, 2])
    with left:
        fig2 = px.scatter(d, x='x', y='y', color='bloc', hover_name='country',
                          title='The same countries on the PCA map (close together = vote alike)')
        fig2.update_layout(height=420, xaxis_title='PCA axis 1', yaxis_title='PCA axis 2')
        st.plotly_chart(fig2, width='stretch')
    with right:
        st.subheader('Who is in each bloc?')
        for bloc_name, group in d.groupby('bloc'):
            with st.expander(f'{bloc_name} ({len(group)} countries)'):
                st.write(', '.join(sorted(group['country'])))


# =================================================================
elif page == 'Compare countries':
    st.title('How often do countries vote the same way?')
    countries = sorted(votes['country'].unique())
    main = st.selectbox('Country', countries, index=countries.index('India'))
    others = st.multiselect('Compare with', [c for c in countries if c != main],
                            default=['United States', 'Russia', 'China'])
    smooth = st.checkbox('Smooth over 3 years', value=True)

    mine = votes[votes['country'] == main][['rcid', 'year', 'vote']]
    lines = []
    for other in others:
        theirs = votes[votes['country'] == other][['rcid', 'vote']]
        both = mine.merge(theirs, on='rcid', suffixes=('_a', '_b'))
        yearly = (both['vote_a'] == both['vote_b']).groupby(both['year']).mean() * 100
        line = yearly.rolling(3, min_periods=1).mean() if smooth else yearly
        lines.append(pd.DataFrame({'year': yearly.index, 'agreement': yearly.values,
                                   'line': line.values, 'with': other}))

    if lines:
        chart = pd.concat(lines)
        fig = px.line(chart, x='year', y='line', color='with',
                      labels={'line': '% same vote', 'year': 'Year', 'with': 'Compared with'},
                      title=f'{main}: share of votes cast the same way')
        fig.update_layout(height=480, yaxis_range=[0, 100])
        st.plotly_chart(fig, width='stretch')

        table = chart.assign(decade=(chart['year'] // 10 * 10).astype(str) + 's')
        st.subheader('Average by decade (%)')
        st.dataframe(table.pivot_table(index='with', columns='decade', values='agreement')
                     .style.format('{:.0f}', na_rep='–'), width='stretch')
    else:
        st.info('Pick at least one country to compare with.')


# =================================================================
elif page == 'Predict a resolution':
    st.title('Predict how countries vote on a resolution')
    st.write('The model was trained on votes from 1980–2009 and never saw these 2010–2019 votes, '
             'so you can compare its predictions with what really happened.')

    year = st.selectbox('Year', sorted(preds['year'].unique(), reverse=True))
    year_res = res[res['rcid'].isin(preds.loc[preds['year'] == year, 'rcid'])].copy()
    year_res['label'] = year_res['descr'].fillna(year_res['short']).str.slice(0, 110)
    # Start on the Cuba embargo vote if it is in this year's list (a nice example of countries changing their vote)
    options = list(year_res['rcid'])
    cuba = year_res[year_res['label'].str.contains('embargo', case=False)]
    start = options.index(cuba['rcid'].iloc[0]) if len(cuba) else 0
    pick = st.selectbox('Resolution', options, index=start,
                        format_func=lambda r: year_res.set_index('rcid').loc[r, 'label'])

    info = res.set_index('rcid').loc[pick]
    st.markdown(f"**Resolution:** {info['descr']}")
    similar = res.set_index('rcid')['descr'].get(info['sim1'], 'none found')
    topic = TOPIC_NAMES.get(info['main_topic'], info['main_topic'])
    st.caption(f"Topic: {topic} · Most similar earlier resolution (similarity {info['similarity']:.2f}): {similar}")

    r = preds[preds['rcid'] == pick].copy()
    accuracy = (r['vote'] == r['predicted']).mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Countries voting', len(r))
    c2.metric('Correct predictions', f'{accuracy:.0f}%')
    c3.metric('Actual yes / no / abstain',
              f"{(r['vote'] == 'yes').sum()} / {(r['vote'] == 'no').sum()} / {(r['vote'] == 'abstain').sum()}")
    c4.metric('Swing countries flagged', int(r['swing_flag'].sum()))

    show = st.radio('Show on the map', ['Predicted vote', 'Actual vote'], horizontal=True)
    col = 'predicted' if show == 'Predicted vote' else 'vote'
    fig = px.choropleth(r.dropna(subset=['iso3']), locations='iso3', color=col, hover_name='country',
                        color_discrete_map=VOTE_COLORS, category_orders={col: ['yes', 'abstain', 'no']})
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0), legend_title_text=show)
    st.plotly_chart(fig, width='stretch')

    st.subheader('Swing countries: most likely to change their usual vote')
    swing = r.sort_values('p_change', ascending=False).head(10)
    st.dataframe(swing[['country', 'vote', 'predicted', 'p_change', 'swing_flag']]
                 .rename(columns={'vote': 'actual', 'p_change': 'chance of change', 'swing_flag': 'flagged'})
                 .style.format({'chance of change': '{:.0%}'}),
                 width='stretch', hide_index=True)

    with st.expander('All countries'):
        st.dataframe(r[['country', 'vote', 'predicted', 'p_yes', 'p_abstain', 'p_no', 'p_change']]
                     .rename(columns={'vote': 'actual'}).sort_values('country')
                     .style.format({c: '{:.0%}' for c in ['p_yes', 'p_abstain', 'p_no', 'p_change']}),
                     width='stretch', hide_index=True)

    st.subheader('Why did the model predict this?')
    country_list = sorted(r['country'])
    default = country_list.index('India') if 'India' in country_list else 0
    who = st.selectbox('Pick a country', country_list, index=default)
    row = r[r['country'] == who].iloc[0]

    model, explainer = load_model()
    class_names = ['no', 'abstain', 'yes']
    k = class_names.index(row['predicted'])
    values = explainer(row[features].to_frame().T.astype(float)).values[0, :, k]
    contrib = pd.DataFrame({'feature': [f'{NICE_NAMES.get(f, f)} = {row[f]:.2f}' for f in features],
                            'value': [row[f] for f in features], 'shap': values})
    contrib = contrib.reindex(contrib['shap'].abs().sort_values(ascending=False).index).head(8)
    contrib['direction'] = np.where(contrib['shap'] > 0, f'pushes towards "{row["predicted"]}"',
                                    f'pushes away from "{row["predicted"]}"')

    st.write(f"**{who}** actually voted **{row['vote']}**. The model predicted **{row['predicted']}** "
             f"(yes {row['p_yes']:.0%}, abstain {row['p_abstain']:.0%}, no {row['p_no']:.0%}).")
    fig = px.bar(contrib.iloc[::-1], x='shap', y='feature', color='direction', orientation='h',
                 hover_data={'value': ':.2f'},
                 color_discrete_sequence=['#1565c0', '#ef6c00'],
                 labels={'shap': 'SHAP contribution', 'feature': ''})
    fig.update_layout(height=380, legend_title_text='')
    st.plotly_chart(fig, width='stretch')
    st.caption('Votes are coded yes = 1, abstain = 0.5, no = 0. Hover over a bar to see the feature value.')


# =================================================================
elif page == 'Model performance':
    st.title('How good are the predictions?')
    st.write('Tested on 163,736 votes from 2010–2019. About 80% of votes are "yes", so accuracy alone is misleading: '
             '**macro F1** gives "no" and "abstain" equal weight.')

    st.subheader('Leaderboard')
    st.dataframe(leaderboard, width='stretch', hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader('XGBoost: confusion matrix')
        fig = px.imshow(confusion.values, x=list(confusion.columns), y=list(confusion.index), text_auto=',',
                        color_continuous_scale='Blues', labels=dict(x='Predicted', y='Actual', color='Votes'))
        fig.update_layout(height=380)
        st.plotly_chart(fig, width='stretch')
    with right:
        st.subheader('Swing flags: precision vs recall')
        known = preds[preds['similar_my_vote'].notna()]
        threshold = st.slider('Flag a country when its chance of change is above', 0.1, 0.9, 0.5, 0.05)
        flagged = known['p_change'] > threshold
        precision = known.loc[flagged, 'changed'].mean() if flagged.any() else 0
        recall = flagged[known['changed']].mean()
        a, b, c = st.columns(3)
        a.metric('Countries flagged', f'{flagged.sum():,}')
        b.metric('Precision', f'{precision:.0%}', help='Of the flagged countries, how many really changed their vote')
        c.metric('Recall', f'{recall:.0%}', help='Of the countries that changed, how many were flagged')
        st.caption(f"Only {known['changed'].mean():.0%} of votes change, so random flags would be right about that often.")
