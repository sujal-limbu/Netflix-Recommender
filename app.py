import streamlit as st
from recommender import load_data, build_similarity, recommend

@st.cache_data
def load_and_prepare_data():
    df = load_data()
    cosine_sim = build_similarity(df)
    return df, cosine_sim

st.title('🎬 Netflix Movie/TV Show Recommender')

try:
    df, cosine_sim = load_and_prepare_data()
    st.write("Data loaded successfully. Number of titles:", len(df))
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

selected_title = st.selectbox('Choose a Netflix title you like:', df['title'].sort_values())

if st.button('Recommend'):
    recommendations = recommend(df, cosine_sim, selected_title)
    if not recommendations:
        st.write("Sorry, no recommendations found.")
    else:
        st.write(f"Because you watched **{selected_title}**, you might also like:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

