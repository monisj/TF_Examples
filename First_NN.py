import tensorflow as TF
import numpy as np
import matplotlib.pyplot as PLT
# print(TF.__version__)
fashion_m=TF.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_m.load_data()

class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
# print(train_images.shape)
# print(len(train_labels))
# print(train_labels)

print(test_images.shape)
print(len(test_labels))
