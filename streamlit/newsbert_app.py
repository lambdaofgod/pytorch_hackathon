import streamlit as st
import pandas as pd
import numpy as np
import tqdm
import os
from operator import itemgetter

import torch
from pytorch_hackathon import rss_feeds, zero_shot_learning, haystack_search
import seaborn as sns

import logging
logging.disable(logging.CRITICAL)


@st.cache(allow_output_mutation=True)
def get_feed_df(rss_feed_urls):
    with st.spinner('Retrieving articles from feeds...'):
        return rss_feeds.get_feed_df(rss_feed_urls)


def get_displayed_df():
    results_csv_path = 'data/zsl_feed_results.csv'
    cached_result_exists = os.path.exists(results_csv_path)
    if cached_result_exists:
        results_df = pd.read_csv(results_csv_path)
    else:
        from utils import streamlit_tqdm
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


@st.cache
def get_retrieved_topic_df(searcher, topic_strings=None, query=None, top_k=20):
    assert not topic_strings is None or not query is None, "Must supply either topics or query"
    if query is not None:
        results = searcher.search(query, top_k=top_k)
    elif topic_strings is not None:
        results = [
            result 
            for topic in topic_strings
            for result in searcher.search(
                "text is about {}".format(topic),
                top_k=top_k
            )
        ]
    return searcher.get_topic_score_df(
        results,
        topic_strings
    ).drop_duplicates(subset='title')


def get_tile_html(row):

    tile_html_template = """
        <div class="card">
        <div class="container">
            <h4><b><a href="{}">{}</a></b></h4>
            <h4>{}</h4>
            <p>{}<p>
        </div>
        </div>
    """
    return tile_html_template.format(row['link'], row['title'], row['date'], row['text'])


def display_wall(selected_df, sort_by, topics, prob):
    display_df = selected_df[selected_df[topics].max(axis=1) > prob/100].sort_values(sort_by, ascending=False)

    st.markdown('## Articles on {}'.format(' or '.join(topics)))

    for __, row in display_df.iterrows():
        st.markdown(get_tile_html(row), unsafe_allow_html=True)


def display_dataframe(selected_df, sort_by, topics, prob):
    cm = sns.light_palette("green", as_cmap=True)
    display_df = selected_df[selected_df[topics].min(axis=1) > prob/100].sort_values(sort_by, ascending=False)
    st.markdown('## Articles on {}'.format(' and '.join(topics)))

    st.table(display_df[display_df[topics]].style.background_gradient(cmap=cm))


def display_data(display_mode, selected_df, sort_by, topics, prob):
    if display_mode == "dataframe":
        display_dataframe(selected_df, sort_by, topics, prob)
    elif display_mode == "wall":
        display_wall(selected_df, sort_by, topics, prob)


def main():
    st.title('NewsBERT')

    # setting up data
    file_url_template = "https://raw.githubusercontent.com/lambdaofgod/pytorch_hackathon/master/data/{}.txt"
    topic_strings_file_url = file_url_template.format('topics')
    rss_feeds_file_url = file_url_template.format('topics')
    topic_strings = list(pd.read_table(topic_strings_file_url, header=None).iloc[:,0].values)
    rss_feed_urls = list(pd.read_table(rss_feeds_file_url, header=None).iloc[:,0].values)
    rss_feed_urls = rss_feeds.rss_feed_urls.copy()

    # most important parameters - changing them reruns the whole app
    model_device = st.sidebar.selectbox("Model device", ["cpu", "cuda"], index=int(torch.cuda.is_available()))
    display_mode = st.sidebar.selectbox("Display mode", ["wall", "dataframe"], index=0)

    feed_df = get_feed_df(rss_feed_urls)
    # we need to copy feed_df so that streamlit doesn't recompute embeddings when feed_df changes 
    searcher = setup_searcher(feed_df.copy(), use_gpu=model_device == 'cuda')

    text_query = st.text_input("Searched phrase", "")
    embedded_query = st.text_input("Searched phrase (embedding search)", "")
    if embedded_query == "":
        embedded_query = None
    topics = st.multiselect('Choose topics', topic_strings, default=[topic_strings[0]])
    prob = st.slider('Min topic confidence', 0, 100, 20)
    selected_df = get_retrieved_topic_df(searcher, topics, embedded_query).reset_index(drop=True)
    if text_query != "":
        selected_df = selected_df[selected_df['text'].str.lower().str.contains(text_query.lower())]

    selected_df['text'] = selected_df['text'].apply(lambda s: s[:1000])
    sort_by = st.selectbox("Sort by", topics)
    display_data(display_mode, selected_df, sort_by, topics, prob)

    
if __name__ == '__main__':
    main()
