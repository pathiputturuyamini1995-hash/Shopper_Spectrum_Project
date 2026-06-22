import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load GMM Model
try:
    gmm_model = joblib.load("best_gmm_model.pkl")
except:
    gmm_model = None

# Sidebar
st.sidebar.title("Shopper Spectrum")

page = st.sidebar.radio(
    "Select Module",
    ["Home", "Customer Segmentation", "Product Recommendation"]
)

# Home Page
if page == "Home":

    st.title("Shopper Spectrum")

    st.write("""
    Customer Segmentation and Product Recommendation System

    Features:
    - Customer Segmentation using GMM Clustering
    - Product Recommendation using Collaborative Filtering
    - Cosine Similarity based Product Matching
    """)

# Customer Segmentation
elif page == "Customer Segmentation":

    st.title("Customer Segmentation")

    recency = st.number_input(
        "Recency (days)",
        min_value=0.0,
        value=30.0
    )

    frequency = st.number_input(
        "Frequency (number of purchases)",
        min_value=0.0,
        value=5.0
    )

    monetary = st.number_input(
        "Monetary (total spend)",
        min_value=0.0,
        value=2500.0
    )

    if st.button("Predict Segment"):

        if gmm_model is not None:

            input_data = [[recency, frequency, monetary]]

            cluster = gmm_model.predict(input_data)[0]

            segment_labels = {
                0: "High-Value",
                1: "Regular",
                2: "Occasional",
                3: "At-Risk"
            }

            result = segment_labels.get(cluster, "Unknown")

            st.write("Predicted Cluster:", cluster)
            st.write("This customer belongs to:", result)

        else:
            st.error("Model file not found")

# Product Recommendation
elif page == "Product Recommendation":

    st.title("Product Recommendation")

    product = st.text_input("Enter Product Name")

    if st.button("Recommend"):

        # Item-Based Collaborative Filtering using Cosine Similarity

        products = [
            "BLUE VINTAGE SPOT BEAKER",
            "PINK VINTAGE SPOT BEAKER",
            "POTTING SHED CANDLE CITRONELLA",
            "POTTING SHED ROSE CANDLE",
            "PANTRY CHOPPING BOARD",
            "RED VINTAGE SPOT MUG"
        ]

        product_matrix = np.array([
            [1,1,0,1,0,0],
            [1,1,1,0,0,0], 
            [0,1,1,1,0,0],
            [1,0,1,1,1,0],
            [0,0,1,1,1,1],
            [0,0,0,1,1,1]
        ])

        similarity = cosine_similarity(product_matrix)

        if product in products:

            idx = products.index(product)

            st.subheader("Top Recommended Products")

            scores = list(enumerate(similarity[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)

            for i, score in scores[1:]:
                st.write(products[i])

        else:
            st.write("Product not found")