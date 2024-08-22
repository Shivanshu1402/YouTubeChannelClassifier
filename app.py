import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.data import find
from nltk.data import LookupError
from nltk.stem import WordNetLemmatizer

# Function to download NLTK resources if not already present
def download_nltk_resources():
    try:
        find('corpora/stopwords.zip')
    except LookupError:
        import nltk
        nltk.download('stopwords')

# Download NLTK resources
download_nltk_resources()

# Load the model
model = tf.keras.models.load_model('youtube_channel_classifier_model.keras')

# Define parameters
paddingLen = 70
vocabSize = 10000
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# Initialize tokenizer (same settings as used during training)
tokenizer = Tokenizer(num_words=vocabSize)

# Function to preprocess input
def preprocess_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stop_words)]
    text = " ".join(text)
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=paddingLen, truncating='post', padding='post')
    return np.array(padded)

# Set up the page configuration
st.set_page_config(page_title="YouTube Channel Classifier", page_icon=":rocket:", layout="wide")

# Add custom CSS
st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        background-image: url('https://upload.wikimedia.org/wikipedia/commons/6/65/YouTube_icon_%282013-2017%29.png');
        background-size: 100px;
        background-position: top right;
        background-repeat: no-repeat;
    }
    .footer {
        background-color: #2c3e50;
        color: white;
        padding: 1rem;
        text-align: center;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .footer .links {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .footer .links a {
        color: #1abc9c;
        text-decoration: none;
        font-size: 1rem;
        display: flex;
        align-items: center;
    }
    .footer .links i {
        margin-right: 0.5rem;
    }
    .footer .links a:hover {
        text-decoration: underline;
    }
    .footer p {
        margin: 0;
    }
    .footer h3 {
        color: #ecf0f1; /* Dark color for headings */
    }
    .main h1, .main h2, .main h3 {
        color: #34495e; /* Dark color for headings */
    }
    </style>
    """, unsafe_allow_html=True)

# Main content
with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.title('YouTube Channel Classifier')
    st.write("## Classify YouTube Video Categories")
    
    user_input = st.text_area("Enter the title and description of the video:")
    
    if st.button("Classify"):
        if user_input:
            processed_input = preprocess_text(user_input)
            prediction = model.predict(processed_input)
            class_labels = ['travel', 'science and technology', 'food', 'manufacturing', 'history', 'art and music']
            predicted_class = class_labels[np.argmax(prediction)]
            st.write(f'### Predicted Category: {predicted_class}')
        else:
            st.write("Please enter some text.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <h3>Developer Information</h3>
        <p>Shivanshu Tripathi</p>
        <p>Phone: 9670439648</p>
        <p>Email: <a href="mailto:shivanshutripathi007@gmail.com">shivanshutripathi007@gmail.com</a></p>
        <div class="links">
            <a href="https://www.linkedin.com/in/shivanshu-tripathi" target="_blank"><i class="fab fa-linkedin"></i>LinkedIn</a>
            <a href="https://www.instagram.com/shivanshu_tripathi" target="_blank"><i class="fab fa-instagram"></i>Instagram</a>
            <a href="https://github.com/shivanshutripathi" target="_blank"><i class="fab fa-github"></i>GitHub</a>
        </div>
        <h3>About</h3>
        <p>This app classifies YouTube video categories based on the title and description provided. Built using TensorFlow and Streamlit.</p>
    </div>
    """, unsafe_allow_html=True)
