import tensorflow as TF
import numpy as np
import matplotlib.pyplot as PLT
import Plot as PL
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
# PLT.figure(figsize=(10,10))
# for image_loop in range(30):
#     PLT.subplot(5,6,image_loop+1)
#     PLT.xticks([])
#     PLT.yticks([])
#     PLT.grid(False)
#     PLT.imshow(train_images[image_loop],cmap=PLT.cm.binary)
#     PLT.xlabel(class_names[train_labels[image_loop]])
# PLT.show()

#Building the model
model=TF.keras.Sequential([
    TF.keras.layers.Flatten(input_shape=(28,28)),
    TF.keras.layers.Dense(128, activation='relu'),
    TF.keras.layers.Dense(10)])
#Run the below line if the code is being executed for the first time
model.compile(optimizer='adam',
                 loss=TF.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=10)

test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)
#print("\nTest Accuracy =",test_acc)

probability_model=TF.keras.Sequential([model,TF.keras.layers.Softmax()])
predictions=probability_model.predict(test_images)
#print(predictions[0])
#print(np.argmax(predictions[0]))
#print(test_labels[0])

# loop=12
# PLT.figure(figsize=(6,3))
# PLT.subplot(1,2,1)
# PL.plot_image(loop,predictions[loop],test_labels,test_images)
# PLT.subplot(1,2,2)
# PL.plot_value_array(loop,predictions[loop],test_labels)
# PLT.show()

# no_rows=5
# no_col=3
# no_images=no_rows*no_col
# PLT.figure(figsize=(2*2*no_col,2*no_rows))
# for loop in range(no_images):
#     PLT.subplot(no_rows,2*no_col,2*loop+1)
#     PL.plot_image(loop,predictions[loop],test_labels,test_images)
#     PLT.subplot(no_rows,2*no_col,2*loop+2)
#     PL.plot_value_array(loop,predictions[loop],test_labels)
# PLT.tight_layout()
# PLT.show()

img=test_images[2]
print(img.shape)

img=(np.expand_dims(img,0))
print(img.shape)

predictions_single=probability_model.predict(img)
print(predictions_single)

PL.plot_value_array(1,predictions_single[0],test_labels)
_=PLT.xticks(range(10),class_names,rotation=45)
#PLT.show()

print(np.argmax(predictions_single[0]))