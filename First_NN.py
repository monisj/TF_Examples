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

# print(test_images.shape)
# print(len(test_labels))

# PLT.figure()
# PLT.imshow(train_images[1])# Changing the index changes the image being viewed 
# PLT.colorbar()
# PLT.grid(False)
# PLT.show()

train_images=train_images/255.0
test_images=test_images/255.0
PLT.figure(figsize=(10,10))
for image_loop in range(30):
    PLT.subplot(5,6,image_loop+1)
    PLT.xticks([])
    PLT.yticks([])
    PLT.grid(False)
    PLT.imshow(train_images[image_loop],cmap=PLT.cm.binary)
    PLT.xlabel(class_names[train_labels[image_loop]])
PLT.show()