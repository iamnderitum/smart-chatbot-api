import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam


# Load preprocessed data
X_train = np.load('preprocessing/X_train.npy')
y_train = np.load('preprocessing/y_train.npy')

"""# Build the neural network model
model = Sequential([
    Dense(128, input_shape=(len(X_train[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(set(y_train)), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=1)

# Save the model
model.save('models/chatbot_model.h5')
print("Model saved as chatbot_model.h5")
"""

# Define the input layer
input_layer = Input(shape=(len(X_train[0]),))

# First Dense later with Batch Normalization and Dropout.
x = Dense(128, activation="relu")(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Second Dense Later with Batch Normalization and Dropout
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Third Dense later with Batch Normalization and Dropout
x = Dense(64, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# Output layer with softmax activation
output_layer = Dense(len(set(y_train)), activation="softmax")(x)

# Create the Model using Functional API
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
optimizer = Adam(learning_rate=0.001) # Fine-tune the learning rate if needed
model.compile(loss="sparse_categorical_crossentropy",  optimizer=optimizer, metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=300, batch_size=16, verbose=1)

# Save the model
model.save("models/chatbot_improvemodel_functional.h5")
print("Model saved as chatbot_improvemodel_functional.h5")