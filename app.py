import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import os

# Check if Graphviz is installed
try:
    import graphviz
except ImportError:
    st.error("Graphviz is not installed. Please install it using 'pip install graphviz pydot' and install the system package.")

# Streamlit App Title
st.title("Neural Network Diagram Generator")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Select Features and Target
    st.write("### Select Input Features and Target Column")
    features = st.multiselect("Select Features (Input)", df.columns)
    target = st.selectbox("Select Target (Output)", df.columns)

    if features and target:
        X = df[features].values
        y = df[target].values

        # Build a Simple Neural Network
        model = Sequential([
            Dense(8, activation='relu', input_shape=(len(features),)),
            Dense(4, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Save the Neural Network Diagram
        image_path = "nn_diagram.png"
        plot_model(model, to_file=image_path, show_shapes=True, show_layer_names=True, dpi=200)

        # Display the diagram in Streamlit
        st.write("### Neural Network Diagram")
        st.image(image_path)
