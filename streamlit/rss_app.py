import streamlit as st
import pandas as pd
import tqdm
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
import elasticsearch
from pytorch_hackathon import zero_shot_learning
import ktrain
import seaborn as sns

cm = sns.light_palette("green", as_cmap=True)
topic_strings = zero_shot_learning.topic_strings.copy()
rss_feed_urls = zero_shot_learning.rss_feed_urls.copy()

#zsl_clf = ktrain.text.ZeroShotClassifier(device='cuda')


@st.cache
def get_feed_df():
    return zero_shot_learning.get_feed_df(rss_feed_urls)


feed_df = get_feed_df()


def get_displayed_df(zsl_clf):
    results_df = zero_shot_learning.get_zero_shot_classification_results_df(
        zsl_clf,
        feed_df['summary'],
        topic_strings
    )
    return feed_df[['title', 'summary']].join(results_df)

st.markdown('# Zero-shot RSS feed article classifier')

display_df = pd.read_csv('feed_topics.csv', index_col='index').reset_index(drop=True)#get_displayed_df(zsl_clf)
display_df['summary'] = display_df['summary'].apply(lambda s: s[:1000])
topics = st.multiselect('Choose topics', topic_strings)

st.markdown('## Articles on {}'.format(', '.join(topics)))

st.table(display_df[display_df[topics].min(axis=1) > 0.5].style.background_gradient(cmap=cm))
