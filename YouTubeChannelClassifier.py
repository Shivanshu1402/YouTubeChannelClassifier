import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import nltk

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
        )
    except RuntimeError as e:
        print(e)

# Data Preparation
df = pd.read_csv("youtube_data.csv")
df = df.dropna()
classes = ['travel', 'science and technology', 'food', 'manufacturing', 'history', 'art and music']

strToInt = {cls: idx for idx, cls in enumerate(classes)}

X = df.drop(labels=["Category", "Video Id"], axis=1)
y = df["Category"]
y = y.map(strToInt)
y = tf.keras.utils.to_categorical(y, num_classes=len(classes))

# Combine title and description
X_text = X["Title"] + " " + X["Description"]

# Preprocessing text
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return " ".join(text)

X_text = X_text.apply(preprocess_text)

# Tokenization and Padding
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_text)
X_sequences = tokenizer.texts_to_sequences(X_text)
X_padded = pad_sequences(X_sequences, maxlen=70)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.arange(len(classes)), y=np.argmax(y, axis=1))
class_weights = dict(enumerate(class_weights))

# Model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=70))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(units=128, return_sequences=True, recurrent_dropout=0.3))
model.add(LSTM(units=128))
model.add(Dense(units=len(classes), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), 
                    class_weight=class_weights, callbacks=[early_stopping])

# Save the model
model.save('youtube_channel_classifier_model.h5')

# Plotting Results
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Loss/Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig('training_results.png')
plt.show()
