import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path='data/netflix_titles.csv'):
    df = pd.read_csv(path)
    for col in ['director', 'cast', 'listed_in']:
        df[col] = df[col].fillna('')
    df['combined_features'] = df.apply(lambda row: row['title'] + ' ' + row['director'] + ' ' + row['cast'] + ' ' + row['listed_in'], axis=1)
    return df

def build_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend(df, cosine_sim, title, top_n=10):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    show_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[show_indices].tolist()
