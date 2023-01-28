import tensorflow as TF
import numpy as np
import matplotlib.pyplot as PLT
class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
def plot_image(loop,predictions_array,true_label,img):
    true_label,img=true_label[loop],img[loop]
    PLT.grid(False)
    PLT.xticks([])
    PLT.yticks([])
    PLT.imshow(img,cmap=PLT.cm.binary)
    prediction_label=np.argmax(predictions_array)
    if prediction_label==true_label:
        color='blue'
    else:
        color='red'
    PLT.xlabel("{} {:2.0f}% ({})".format(class_names[prediction_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]),
                                        color=color)
def plot_value_array(loop,predictions_array,true_label):
    true_label=true_label[loop]
    PLT.grid(False)
    PLT.xticks(range(10))
    PLT.yticks([])
    thisplot=PLT.bar(range(10),predictions_array,color="#777777")
    PLT.ylim([0,1])
    predicted_label=np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')