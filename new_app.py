# app.py
import streamlit as st
import numpy as np
from model import FeatureExtractor, SimilarityCalculator, ExplainabilityEngine, UserFeedback

# Initializing Model Components
feature_extractor = FeatureExtractor()
similarity_calculator = SimilarityCalculator()
user_feedback = UserFeedback()
explainability_engine = ExplainabilityEngine(feature_extractor.model)

# Database
#path = "Database/"
database_images = ["1.jpeg", "2.png", "3.png", "4.png", "5.jpg", "6.jpg"]

# Extract features from the database images (precompute features)
# database_features = np.array([feature_extractor.extract_features( path + img) for img in database_images])
database_features = np.array([feature_extractor.extract_features(img) for img in database_images])

# Initialize SHAP explainer with a background dataset
# Use a small subset of the dataset as background data (in this case, all available images)
background_data = database_features
explainability_engine.create_shap_explainer(background_data)

# Streamlit Interface
st.title("Explainable Interactive CBIR System")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Extract features from the uploaded image
    query_features = feature_extractor.extract_features(uploaded_file)

    # Adjust Feature Weights
    st.subheader("Adjust Feature Weights")
    color_weight = st.slider("Color Weight", 0.0, 1.0, 0.5)
    texture_weight = st.slider("Texture Weight", 0.0, 1.0, 0.5)
    shape_weight = st.slider("Shape Weight", 0.0, 1.0, 0.5)

    # Normalize weights
    total_weight = color_weight + texture_weight + shape_weight
    weights = np.array([color_weight, texture_weight, shape_weight]) / total_weight

    # Calculate similarities
    similarities = similarity_calculator.compute_weighted_similarity(query_features, database_features, weights)
    top_indices = np.argsort(similarities[0])[::-1][:3]

    # Display results and explanations
    for idx in top_indices:
        img_path = database_images[idx]
        st.image(img_path, caption=f'Retrieved Image {idx+1}', use_column_width=True)
        
        # Preprocess the retrieved image for explainability
        img_array = feature_extractor.preprocess(img_path)
        
        # Grad-CAM Heatmap
        cam = explainability_engine.grad_cam(img_array, layer_name='conv5_block3_out')
        st.image(cam, caption="Grad-CAM Heatmap", use_column_width=True)

        # SHAP Explanation
        #st.subheader(f"SHAP Explanation for Image {idx+1}")
        #shap_values = explainability_engine.explain_with_shap(img_array)
        #if shap_values:
        #    st.image(shap_values, caption="SHAP Explanation", use_column_width=True)

    # Collect user feedback
    st.subheader("Provide Feedback")
    feedback = st.text_input("Feedback on the retrieved images:")
    if st.button("Submit Feedback"):
        user_feedback.collect_feedback(feedback)
        st.write("Feedback submitted!")
        user_feedback.update_model_based_on_feedback()
