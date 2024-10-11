import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load preprocessed data
X_train = np.load('preprocessing/X_train.npy')
y_train = np.load('preprocessing/y_train.npy')

# Build the neural network model
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
