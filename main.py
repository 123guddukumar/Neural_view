import streamlit as st
import pandas as pd
import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt
import io
import tensorflow as tf
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches  # Import patches for Ellipse
import time
import os
import logging
from PIL import Image, ImageDraw

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to create an impressive neural network diagram like the provided image
def create_impressive_nn_diagram(input_size, hidden_layers, output_size):
    fig, ax = plt.subplots(figsize=(12, 6))  # Larger figure for clarity
    plt.title("Neural Network Architecture", fontsize=16, pad=20, color='black')
    
    # Positions for layers (vertical alignment for cleaner connections)
    layer_positions = []
    current_x = 0.1
    layer_width = 0.8 / (len(hidden_layers) + 2)  # Spread layers evenly with more space
    
    # Colors matching the image (skyblue for input, purple for first hidden, limegreen for second hidden, orange for output)
    colors = ['skyblue', 'purple', 'limegreen', 'orange']
    
    # Input layer (skyblue ovals)
    layer_positions.append(current_x)
    for i in range(input_size):
        y = 0.9 - (i * 0.8 / max(input_size - 1, 1))
        # Use Ellipse from matplotlib.patches
        ellipse = patches.Ellipse((current_x, y), 0.1, 0.06, color=colors[0], alpha=0.8, zorder=10)  # Wider ovals
        ax.add_artist(ellipse)
        ax.text(current_x, y, f'X{i}', ha='center', va='center', fontsize=12, color='white', 
                bbox=dict(facecolor='black', alpha=0.5))  # Clear text with black background
    
    # Hidden layers (purple and limegreen ovals)
    for layer_idx, units in enumerate(hidden_layers):
        current_x += layer_width
        layer_positions.append(current_x)
        color_idx = 1 if layer_idx == 0 else 2  # Purple for first hidden, limegreen for second
        for j in range(units):
            y = 0.9 - (j * 0.8 / max(units - 1, 1))
            ellipse = patches.Ellipse((current_x, y), 0.1, 0.06, color=colors[color_idx], alpha=0.8, zorder=10)
            ax.add_artist(ellipse)
            ax.text(current_x, y, f'H{layer_idx}{j}', ha='center', va='center', fontsize=12, color='white', 
                    bbox=dict(facecolor='black', alpha=0.5))
    
    # Output layer (orange oval)
    current_x += layer_width
    layer_positions.append(current_x)
    for k in range(output_size):
        y = 0.9 - (k * 0.8 / max(output_size - 1, 1))
        ellipse = patches.Ellipse((current_x, y), 0.1, 0.06, color=colors[3], alpha=0.8, zorder=10)
        ax.add_artist(ellipse)
        ax.text(current_x, y, f'Y{k}', ha='center', va='center', fontsize=12, color='white', 
                bbox=dict(facecolor='black', alpha=0.5))
    
    # Draw connections (straight purple lines like the image)
    for i in range(len(layer_positions) - 1):
        prev_layer_x = layer_positions[i]
        next_layer_x = layer_positions[i + 1]
        prev_y_coords = [0.9 - (j * 0.8 / max(input_size if i == 0 else hidden_layers[i-1 if i > 0 else 0] - 1, 1)) 
                         for j in range(input_size if i == 0 else hidden_layers[i-1 if i > 0 else 0])]
        next_y_coords = [0.9 - (j * 0.8 / max(hidden_layers[i] if i < len(hidden_layers) else output_size - 1, 1)) 
                         for j in range(hidden_layers[i] if i < len(hidden_layers) else output_size)]
        
        for py in prev_y_coords:
            for ny in next_y_coords:
                # Draw straight lines connecting the centers of ovals
                ax.plot([prev_layer_x, next_layer_x], [py, ny], color='purple', linestyle='-', alpha=0.5, linewidth=0.5, zorder=5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)  # Higher DPI for clarity
    buf.seek(0)
    plt.close()
    return buf

# Function to simulate data processing (live thinking) with increasing oval sizes and clear text
def simulate_data_processing(X, model, hidden_layers):
    fig, ax = plt.subplots(figsize=(14, 8))  # Larger figure for animation
    plt.title("Simulating Data Processing Through Neural Network", fontsize=16, pad=20, color='black')
    
    # Initialize layers for visualization
    layer_values = [X[0]]  # Start with the first data point
    for units in hidden_layers:
        layer_values.append(np.zeros(units))
    layer_values.append(np.zeros(1))  # Output layer
    
    # Colors matching the diagram (skyblue for input, purple/green for hidden, orange for output)
    colors = ['skyblue', 'purple', 'limegreen', 'orange']
    
    # Plot initial state with increasing oval sizes and clear text
    def update(frame):
        ax.clear()
        plt.title(f"Simulating Data Processing (Step {frame + 1})", fontsize=16, color='black')
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')  # Light grid for context
        
        # Simulate data flow with increasing oval sizes
        for layer_idx in range(len(hidden_layers) + 1):
            if layer_idx == 0:
                values = layer_values[0]  # Input
                color = colors[0]
                layer_label = 'Input Layer'
            elif layer_idx < len(hidden_layers):
                # Simulate hidden layer activation (simple linear pass for demo)
                values = np.tanh(np.dot(layer_values[layer_idx-1], np.random.randn(len(layer_values[layer_idx-1]), len(layer_values[layer_idx]))))
                layer_values[layer_idx] = values
                color = colors[1] if layer_idx == 1 else colors[2]
                layer_label = f'Hidden Layer {layer_idx}'
            else:
                # Output layer
                values = np.tanh(np.dot(layer_values[-2], np.random.randn(len(layer_values[-2]), 1)))
                layer_values[-1] = values
                color = colors[3]
                layer_label = 'Output Layer'
            
            # Position nodes vertically with more spacing
            y_positions = np.linspace(0.9, 0.1, len(values))
            # Increase oval size based on frame (simulation progress)
            oval_width = 0.1 + (frame * 0.02)  # Gradually increase width up to 0.4
            oval_height = 0.06 + (frame * 0.012)  # Gradually increase height up to 0.24
            if oval_width > 0.4:  # Cap size for readability
                oval_width = 0.4
            if oval_height > 0.24:
                oval_height = 0.24
            
            for i, val in enumerate(values):
                ellipse = patches.Ellipse((layer_idx * 0.4, y_positions[i]), oval_width, oval_height, color=color, alpha=0.8, zorder=10)
                ax.add_artist(ellipse)
                ax.text(layer_idx * 0.4, y_positions[i], f'{val:.2f}', ha='center', va='center', 
                        fontsize=14, color='white', bbox=dict(facecolor='black', alpha=0.7))  # Larger, clear text
            
            # Label the layer
            ax.text(layer_idx * 0.4, 1.05, layer_label, ha='center', va='center', fontsize=14, color='black')
        
        ax.set_xlim(-0.2, (len(hidden_layers) + 2) * 0.4)
        ax.set_ylim(0, 1.2)
        ax.axis('off')
    
    # Create animation with slower speed for clarity
    try:
        ani = FuncAnimation(fig, update, frames=15, interval=800, repeat=False)  # More frames, longer interval
        
        # Save animation to a temporary file
        temp_file = "live_processing.gif"
        ani.save(temp_file, writer='pillow', fps=1)  # Slower frame rate for clarity
        logger.info(f"Animation saved to {temp_file}")
        
        # Read the file as bytes
        with open(temp_file, 'rb') as f:
            gif_bytes = f.read()
        
        # Clean up the temporary file
        os.remove(temp_file)
        logger.info("Temporary file removed")
        
        return gif_bytes
    except Exception as e:
        logger.error(f"Error in simulating data processing: {str(e)}")
        raise

# Function to crop image into a circle
def crop_to_circle(image_path):
    try:
        # Open the image
        img = Image.open(image_path).convert('RGBA')
        width, height = img.size
        
        # Create a circular mask
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        diameter = min(width, height)
        center = (width // 2, height // 2)
        draw.ellipse((center[0] - diameter // 2, center[1] - diameter // 2, 
                      center[0] + diameter // 2, center[1] + diameter // 2), fill=255)
        
        # Apply the mask to the image
        result = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        result.paste(img, (0, 0), mask)
        
        # Resize to a square for circular display (optional, adjust size as needed)
        size = min(width, height)
        result = result.resize((size, size), Image.Resampling.LANCZOS)
        
        # Save to a buffer for Streamlit
        buf = io.BytesIO()
        result.save(buf, format='PNG')
        buf.seek(0)
        return buf
    except Exception as e:
        logger.error(f"Error cropping image to circle: {str(e)}")
        return None

# Streamlit app
st.title("Impressive Neural Network Diagram with Clear Live Data Processing")

# Sidebar with About Developer (fixed content with circular image)
with st.sidebar:
    st.header("About Developer")
    st.write("**Name:** Guddu Kumar")
    st.write("**Role:** Full Stack Developer")
    st.write("""
        Guddu Kumar is a passionate Full Stack Developer with expertise in building end-to-end web and AI applications. 
        With a strong foundation in both front-end and back-end technologies, Guddu excels in creating scalable, user-friendly solutions. 
        He is particularly interested in integrating machine learning and neural networks into real-world projects, making complex technologies accessible and impactful.
    """)
    
    # Add fixed developer image as a circle (replace with your actual image path or URL)
    developer_image_path = "me.jpg"  # Replace with your image path or URL, e.g., "https://example.com/developer.jpg"
    circular_image = crop_to_circle(developer_image_path)
    if circular_image:
        st.image(circular_image, caption="Guddu Kumar - Full Stack Developer", use_column_width=True)
    else:
        st.error("Failed to load or crop the developer image. Please check the image path or URL.")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Assume the last column is the target (output), rest are features (input)
    input_size = df.shape[1] - 1  # Number of features
    output_size = 1  # Assuming binary or single output for simplicity

    # Define a simple neural network structure
    hidden_layers = [4, 3]  # Example: 2 hidden layers with 4 and 3 neurons

    st.write(f"Detected {input_size} input features and assuming 1 output.")

    # Generate the impressive neural network diagram
    nn_diagram = create_impressive_nn_diagram(input_size, hidden_layers, output_size)
    st.image(nn_diagram, caption='Neural Network Architecture', use_column_width=True)

    # Optional: Train a simple model and show "live thinking" (predictions)
    if st.button("Run Predictions and Show Live Processing"):
        # Prepare data
        X = df.iloc[:, :-1]  # Features
        y = df.iloc[:, -1]   # Target

        # Convert categorical data to numeric (if any) and handle missing values
        X = pd.get_dummies(X)  # One-hot encode categorical columns
        X = X.fillna(0)        # Replace NaN with 0
        y = pd.get_dummies(y) if y.dtype == 'object' else y  # Encode target if categorical
        y = y.fillna(0)        # Replace NaN in target

        # Ensure numeric data and convert to NumPy arrays with float32 type
        X = X.astype('float32').values
        y = y.astype('float32').values if len(y.shape) > 1 else y.values  # Handle multi-column y

        # Adjust output_size if y has multiple columns (e.g., one-hot encoded)
        output_size = y.shape[1] if len(y.shape) > 1 else 1

        # Simple TensorFlow model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_layers[0], activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dense(hidden_layers[1], activation='relu'),
            tf.keras.layers.Dense(output_size, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train briefly for demo
        try:
            model.fit(X, y, epochs=5, verbose=0)
            predictions = model.predict(X[:5])  # Predict on first 5 rows
            st.write("Sample Predictions (first 5 rows):")
            st.write(predictions)

            # Show live data processing simulation
            try:
                processing_animation = simulate_data_processing(X, model, hidden_layers)
                st.image(processing_animation, caption='Clear Live Data Processing Simulation', use_column_width=True)
                logger.info("Live processing animation displayed successfully")
            except Exception as e:
                st.error(f"Error displaying live processing: {str(e)}")
                logger.error(f"Error displaying live processing: {str(e)}")

        except Exception as e:
            st.error(f"Error during training/prediction: {str(e)}")
            logger.error(f"Error during training/prediction: {str(e)}")

# Run instructions
st.sidebar.write("**Contact US:** [Instagram](https://www.instagram.com/123divyansuverma)")