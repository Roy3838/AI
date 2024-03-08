
import json
import os
import sys
from tqdm import tqdm

def load_jsonl_to_memory(filepath, fraction=4):
    # Determine the total number of lines to calculate the size of the fraction
    with open(filepath, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)
    
    # Calculate the number of lines to process based on the fraction
    lines_to_process = total_lines // fraction
    
    # Preallocate the list with None values for the fraction of data
    data = [None] * lines_to_process
    
    with open(filepath, 'r', encoding='utf-8') as file:
        processed_lines = 0  # Keep track of how many lines have been processed
        for index, line in enumerate(tqdm(file, total=total_lines, desc="Processing")):
            if index % fraction == 0:  # Process only every fraction-th line
                # Parse the JSON content from the line and add it to the data list
                data[processed_lines] = json.loads(line)
                processed_lines += 1
                if processed_lines >= lines_to_process:
                    break  # Stop if we've processed the intended number of lines
    
    return data

data = load_jsonl_to_memory('/mnt/datassd/processed_file.jsonl')

# |%%--%%| <iNSJjndvSD|oWVm17uamQ>

# data in GB
sys.getsizeof(data) / 1024**2

# |%%--%%| <oWVm17uamQ|jeNktIzR1s>

len(data)

# |%%--%%| <jeNktIzR1s|hTck5MLDRM>

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

# Assuming 'data' is your list of dictionaries
titles = [item['title'] for item in data]
view_counts = np.array([item['view_count'] for item in data])

# Parameters for tokenization and padding
vocab_size = 10000  # Adjust based on your dataset
max_length = 100  # Adjust based on the length of your titles
padding_type = 'post'
trunc_type = 'post'

# Initialize and fit the tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(titles)

# Convert titles to sequences and pad them
sequences = tokenizer.texts_to_sequences(titles)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Split the data into training and testing sets
X_train, X_test, y_train_n, y_test_n = train_test_split(padded_sequences, view_counts, test_size=0.2, random_state=42)


# Convert to log scale for normal distribution of data
y_test = np.log(y_test_n)
y_test = np.where(y_test == -np.inf, 0, y_test)

y_train = np.log(y_train_n)
y_train = np.where(y_train == -np.inf, 0, y_train)


# |%%--%%| <hTck5MLDRM|UZ5jTpSq96>

import numpy as np
import keras
from keras import layers
import keras_nlp

vocab_size = 10000  # Adjust based on your vocabulary size
embedding_dim = 256
max_length = 100  # Adjust based on your titles' maximum length
num_heads = 8  # Number of attention heads in the Transformer encoder
intermediate_dim = 512  # Dimensionality of the encoder's intermediate (feed-forward) layer

# Define input layer
inputs = keras.Input(shape=(max_length,), dtype='int64')

# Token and position embedding layer
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=vocab_size,
    sequence_length=max_length,
    embedding_dim=embedding_dim,
)
x = embedding_layer(inputs)

# Transformer encoder layer
encoder = keras_nlp.layers.TransformerEncoder(
    num_heads=num_heads,
    intermediate_dim=intermediate_dim,
    activation='relu',
    dropout=0.1,
)
x = encoder(x)

# Since we're working on a regression task, a GlobalMaxPooling1D layer is used to reduce the sequence dimension
x = layers.GlobalMaxPooling1D()(x)

# Additional dense layers for further processing
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(1, activation='linear')(x)  # Linear activation for a regression task

# Compile the model
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error')

model.summary()


# |%%--%%| <UZ5jTpSq96|nlXeHFPnD6>

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import Adam


vocab_size = 10000  # Adjust based on your vocabulary size
embedding_dim = 256
max_length = 100  # Adjust based on your titles' maximum length

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=4),
    LSTM(128, return_sequences=True),
    GlobalMaxPooling1D(),
    Dense(256, activation='relu'),
    Dense(1, activation='linear')  # Linear activation for a regression task
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')

model.summary()


# |%%--%%| <nlXeHFPnD6|sogp3wmkqN>

del data


# |%%--%%| <sogp3wmkqN|A5CwqIiqkS>

# Assuming X_train, y_train are your training data and labels, respectively
model.fit(X_train, y_train, validation_split=0.1, epochs=15, batch_size=32)


# |%%--%%| <A5CwqIiqkS|NjVou6EDo0>

model.save('YT_T30_log.keras')

# |%%--%%| <NjVou6EDo0|u4VHnlqHHe>

import tensorflow as tf
import keras
#del X_train, y_train

model = tf.keras.models.load_model('YT_T15_log.keras')

# |%%--%%| <u4VHnlqHHe|Fy2WrIxI2U>

# Predict view counts for the test set
predictions = model.predict(X_test, verbose=1)

# Optionally, compare the first few predictions to the actual view counts
for i in range(10):  # Display first 10 predictions
    print(f"Predicted view count: {predictions[i]}, Actual view count: {y_test[i]}")


# |%%--%%| <Fy2WrIxI2U|1b8b8x0TCu>

import matplotlib.pyplot as plt



# Plot the predicted vs. actual view counts

plt.scatter(y_test, predictions, alpha=0.4)
plt.xlabel('Actual View Count')
plt.ylabel('Predicted View Count')
plt.savefig('t30_log.png')


# |%%--%%| <1b8b8x0TCu|133QJ2PHON>

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test, verbose=1)

# If you have specified any metrics when compiling the model, they will also be returned
# Example: model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# Then you can unpack the results as follows:
# loss, mae = model.evaluate(X_test, y_test, verbose=1)

print(f"Test Loss: {loss}")
# If applicable: print(f"Test MAE: {mae}")


# |%%--%%| <133QJ2PHON|OCDRXaLHZE>

model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32)

# |%%--%%| <OCDRXaLHZE|LKDv6pvBDA>

model.save('YT_model20epochs.keras')

# |%%--%%| <LKDv6pvBDA|0bZ9b8knsf>

# Predict view counts for the test set
predictions = model.predict(X_test, verbose=1)

# Optionally, compare the first few predictions to the actual view counts
for i in range(10):  # Display first 10 predictions
    print(f"Predicted view count: {predictions[i]}, Actual view count: {y_test[i]}")

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test, verbose=1)

print(f"Test Loss: {loss}")

# |%%--%%| <0bZ9b8knsf|9IRxPNoVbP>

model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32)
