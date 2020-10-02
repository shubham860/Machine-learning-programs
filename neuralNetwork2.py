import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1,
                                                input_shape=[1])])  # Your Code Here#
model.compile(optimizer='sgd', loss='mean_squared_error')  # Your Code Here#
xs = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9], dtype=int)  # Your Code Here#
ys = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5],
              dtype=float)  # Your Code Here#
model.fit(xs, ys, epochs=100)  # Your Code here#
print('Housing Price for house with 7 bedrooms: ', model.predict([7.0])*100)
