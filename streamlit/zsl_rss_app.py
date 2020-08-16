import streamlit as st
import pandas as pd
import numpy as np
import tqdm
import os
from operator import itemgetter

import torch
from pytorch_hackathon import rss_feeds, zero_shot_learning, haystack_search
import seaborn as sns
from utils import streamlit_tqdm

st.title('Zero-shot RSS feed article classifier')

cm = sns.light_palette("green", as_cmap=True)
topic_strings = list(pd.read_table('data/topics.txt', header=None).iloc[:,0].values)
rss_feed_urls = list(pd.read_table('data/feeds.txt', header=None).iloc[:,0].values)
rss_feed_urls = rss_feeds.rss_feed_urls.copy()


model_device = st.selectbox("Model device", ["cpu", "cuda"], index=int(torch.cuda.is_available()))


@st.cache(allow_output_mutation=True)
def get_feed_df():
    with st.spinner('Retrieving articles from feeds...'):
        return rss_feeds.get_feed_df(rss_feed_urls)


feed_df = get_feed_df()


def get_displayed_df():
    results_csv_path = 'data/zsl_feed_results.csv'
    cached_result_exists = os.path.exists(results_csv_path)
    if cached_result_exists:
        results_df = pd.read_csv(results_csv_path)
    else:
        import ktrain
        with st.spinner('No precomputed topics found, running zero-shot learning'):
            zsl_clf = ktrain.text.ZeroShotClassifier(device=model_device)
            results_df = zero_shot_learning.get_zero_shot_classification_results_df(
                zsl_clf,
                feed_df['text'],
                topic_strings,
                progbar_wrapper=streamlit_tqdm
            )
            results_df.to_csv(results_csv_path, index=False)
    return feed_df[['title', 'text']].join(results_df)


@st.cache(allow_output_mutation=True)
def setup_searcher(feed_df, use_gpu, model_name="deepset/sentence_bert"):
    with st.spinner('No precomputed topics found, running zero-shot learning...'):
        searcher = haystack_search.Searcher(model_name, 'text', use_gpu=use_gpu)
        searcher.add_texts(feed_df)
    return searcher 


# we need to copy feed_df so that streamlit doesn't recompute embeddings when feed_df changes 
searcher = setup_searcher(feed_df, use_gpu=model_device == 'cuda') 


@st.cache
def get_retrieved_df(topic_strings):
    results = [
        result 
        for topic in topic_strings
        for result in searcher.retriever.retrieve(
            "text is about {}".format(topic)
        )
    ]
    return searcher.get_topic_score_df(
        results,
        topic_strings
    ).drop_duplicates(subset='title')
    

selected_df = get_retrieved_df(topic_strings).reset_index(drop=True)
selected_df['text'] = selected_df['text'].apply(lambda s: s[:1000])
topics = st.multiselect('Choose topics', topic_strings, default=[topic_strings[0]])
sort_by = st.selectbox("Sort by", topics)
display_df = selected_df[selected_df[topics].min(axis=1) > 0.5].sort_values(sort_by, ascending=False)

st.markdown('## Articles on {}'.format(', '.join(topics)))

st.table(display_df[display_df[topics].min(axis=1) > 0.5].style.background_gradient(cmap=cm))
