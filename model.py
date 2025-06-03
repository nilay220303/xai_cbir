import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import shap

# Feature Extractor Class
class FeatureExtractor:
    def __init__(self):
        # ResNet50 pre-trained model, without the top layers
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    def preprocess(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        return img_data

    def extract_features(self, img_path):
        img_data = self.preprocess(img_path)
        features = self.model.predict(img_data)
        return features.flatten()
        


# Similarity Calculator Class
class SimilarityCalculator:
    def __init__(self):
        pass

    def compute_weighted_similarity(self, query_features, database_features, weights):
        # Assuming weights are applied uniformly for now
        expanded_weights = np.ones(query_features.shape) * np.mean(weights)
        
        # Ensure query_features and database_features are 2D
        query_features = query_features.reshape(1, -1)  # Shape (1, 2048)
        database_features = database_features.reshape(-1, query_features.shape[1])  # Shape (n, 2048)
        
        weighted_query_features = query_features * expanded_weights
        weighted_database_features = database_features * expanded_weights
        
        similarities = cosine_similarity(weighted_query_features, weighted_database_features)
        return similarities


# Explainability Engine Class
class ExplainabilityEngine:
    def __init__(self, model):
        self.model = model
        self.explainer = None

    def create_shap_explainer(self, background_data):
        if(background_data is None or len(background_data) == 0):
            raise ValueError("Invalid background data for SHAP explainer")
        
        self.explainer = shap.GradientExplainer(self.model, background_data)
        
        

    def grad_cam(self, img_array, layer_name):
        grad_model = tf.keras.models.Model([self.model.inputs], [self.model.get_layer(layer_name).output, self.model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if predictions is None or len(predictions) == 0:
                raise ValueError("Invalid model predictions")
            loss = predictions[:, np.argmax(predictions[0])]
        grads = tape.gradient(loss, conv_outputs)[0]
        output = conv_outputs[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.dot(output, weights)
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        return cam

    def explain_with_shap(self, img_array):
        if(self.explainer is None):
            raise ValueError("SHAP explainer has not been initialized. Call create_shap_explainer() with valid background data first.")
        
        if(img_array is None or img_array.size == 0):
            raise ValueError("Invalid input: img_array is None or empty")

        prediction = self.model.predict(img_array)
        if prediction is None or len(prediction) == 0:
            raise ValueError("Invalid model prediction")

        shap_values = self.explainer.shap_values(img_array)
        return shap_values

    def display_shap(self, img_array):
        st.subheader("SHAP Explanations")
        shap_values = self.explain_with_shap(img_array)
        shap.image_plot(shap_values, img_array)



# User Feedback Class
class UserFeedback:
    def __init__(self):
        self.feedback_data = []

    def collect_feedback(self, feedback):
        self.feedback_data.append(feedback)

    def update_model_based_on_feedback(self):
        if(len(self.feedback_data) > 0):
            print(f"Updating model based on Feedback!")
        else:
            print("No feedback collected yet.")

