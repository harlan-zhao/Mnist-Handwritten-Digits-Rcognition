import tensorflow as tf
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist #28x28 images of hand-written digit 0-9
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# CNN model ---sequential

new_model = tf.keras.models.load_model('1.0_mnist')
prediction = new_model.predict(x_test)
print(len(prediction))
import numpy as np
print(np.argmax(prediction[100]))
plt.imshow(x_test[100])
plt.show()