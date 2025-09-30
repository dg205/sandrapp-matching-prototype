import streamlit as st
import pandas as pd
from model_matching import encode_pair, predict_match_score

st.title("🧑‍🤝‍🧑 SANDRApp Matching Prototype")
st.write("Enter user details below to test matching scores.")

# --- User 1 ---
st.header("User 1")
age1 = st.number_input("Age (User 1)", min_value=18, max_value=100, value=25)
city1 = st.text_input("City (User 1)", "atl")
religion1 = st.text_input("Religion (User 1)", "christian")
hobbies1 = st.text_input("Hobbies (comma-separated, User 1)", "hiking,music").lower().split(",")

# --- User 2 ---
st.header("User 2")
age2 = st.number_input("Age (User 2)", min_value=18, max_value=100, value=27)
city2 = st.text_input("City (User 2)", "chi")
religion2 = st.text_input("Religion (User 2)", "muslim")
hobbies2 = st.text_input("Hobbies (comma-separated, User 2)", "cooking,reading").lower().split(",")

# --- Run prediction ---
if st.button("Run Match"):
    u1 = {"age": age1, "city": city1, "religion": religion1, "interests": set(hobbies1)}
    u2 = {"age": age2, "city": city2, "religion": religion2, "interests": set(hobbies2)}

    features = encode_pair(u1, u2)
    score = predict_match_score(features)

    st.success(f"Match Score: {score}")
    st.subheader("Feature Breakdown")
    st.json(features)
