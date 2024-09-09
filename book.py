# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 00:07:36 2024

@author: balak
"""

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load data
data1 = pd.read_csv('data.csv')  # Replace with your actual file path

# Prepare books dataframe
books = data1[['Book-Title', 'Book-Author', 'Publisher', 'ISBN', 'Total-Ratings']].drop_duplicates()
books['combined_features'] = books['Book-Title'] + " " + books['Book-Author'] + " " + books['Publisher']
books['combined_features'] = books['combined_features'].fillna('')

# Compute the TF-IDF matrix and cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Example user-item matrix for collaborative filtering
user_item_matrix = data1.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)
user_item_matrix_sparse = csr_matrix(user_item_matrix.values)
model = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute', n_jobs=-1)
model.fit(user_item_matrix_sparse)

# Functions to get recommendations
def get_content_based_recommendations(book_title, top_n=5):
    idx = books[books['Book-Title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return books[['Book-Title', 'ISBN']].iloc[book_indices]

def get_collaborative_recommendations(book_isbn, top_n=5):
    idx = user_item_matrix_sparse[:, books[books['ISBN'] == book_isbn].index[0]]
    distances, indices = model.kneighbors(idx, n_neighbors=top_n+1)
    recommendations = [(books.iloc[i]['Book-Title'], distances[0][j]) for j, i in enumerate(indices[0]) if i != books[books['ISBN'] == book_isbn].index[0]]
    return pd.DataFrame(recommendations, columns=['Book-Title', 'Score'])

def get_popularity_based_recommendations(top_n=5):
    return books[['Book-Title', 'ISBN']].sort_values(by='Total-Ratings', ascending=False).head(top_n)

# Streamlit app
st.title('Book Recommendation System')

# Input from user
book_title = st.text_input('Enter a Book Title:', '')

if book_title:
    # Get recommendations
    content_based_recs = get_content_based_recommendations(book_title)
    book_isbn = books[books['Book-Title'] == book_title]['ISBN'].values[0]
    collaborative_recs = get_collaborative_recommendations(book_isbn)
    popularity_recs = get_popularity_based_recommendations()
    
    # Display recommendations
    st.subheader('Content-Based Recommendations')
    st.write(content_based_recs)

    st.subheader('Collaborative Filtering Recommendations')
    st.write(collaborative_recs)

    st.subheader('Popularity-Based Recommendations')
    st.write(popularity_recs)