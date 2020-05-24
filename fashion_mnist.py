# Larger CNN for the MNIST Dataset
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
k=X_test
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define the larger model
def larger_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(15, (3, 3), activation='relu'))
    #model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D())



    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print(model.summary())
print("ERROR: %.2f%%" % (100-scores[1]*100))
print("MODEL ACCURACY: %.2f%%" % (scores[1]*100))
acc=scores[1]*100
file=open("file.txt",'w')
file.write(str(acc))
file.close()
'''p=model.predict(X_test)
k=int(input("Enter the index = "))
pi=np.argmax(p[k])
if pi==0:
    print("Provide test image is T-shirt or Top")
elif pi==1:
    print("Provide test image is Trouser")
elif pi==2:
    print("Provide test image is Pullover")
elif pi==3:
    print("Provide test image is Dress")
elif pi==4:
    print("Provide test image is Coat")
elif pi==5:
    print("Provide test image is Sandal")
elif pi==6:
    print("Provide test image is Shirt")
elif pi==7:
    print("Provide test image is Sneaker")
elif pi==8:
    print("Provide test image is Bag")
elif pi==9:
    print("Provide test image is Ankle boot")
plt.figure()
plt.imshow(np.squeeze(X_test[k]))
plt.show()'''    
