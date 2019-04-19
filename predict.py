import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tkinter import messagebox
import tkinter

mnist = tf.keras.datasets.mnist #28x28 images of hand-written digit 0-9
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
model = tf.keras.models.load_model('first_try')
predictions = model.predict(x_test)
plt.imshow(x_test[666], cmap=plt.cm.binary)
plt.show()

root=tkinter.Tk()
root.withdraw()
messagebox.showinfo("Prediction","The number on the picture is {}".format(np.argmax(predictions[666])))

