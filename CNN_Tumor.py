from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

def readDirectoryAndLabel(directory,label):                                                   #return list that in name of files in given directory
    imageFiles  = [cv2.imread(directory+"/"+f) for f in listdir(directory) if isfile(join(directory, f))]   #image name added to list
    imageLabels = []                                                                                        #keep label of image
    for i in range(len(imageFiles)):
        imageLabels.extend([label])
    return imageFiles,imageLabels


if __name__ == "__main__":



    tumor = "img/Tumor"                                                     #File divided two folders test and train
    normal = "img/Normal"

    train, trainLabel = readDirectoryAndLabel(tumor + "/train", "tumor")     #gets train tumor files and label
    x,y =  readDirectoryAndLabel(normal + "/train","normal")                 #gets train normal files and label
    train.extend(x)                                                          #train files merged
    trainLabel.extend(y)                                                     #train label merged


    test, testLabel = readDirectoryAndLabel(tumor + "/test", "tumor")       #gets test tumor files and label
    x, y = readDirectoryAndLabel(normal + "/test", "normal")                #gets test normal files and label
    test.extend(x)                                                          #test files merged
    testLabel.extend(y)                                                     #test label merged

    train = np.array(train)                                                 #typecast list to numpy array
    trainLabel = np.array(trainLabel)
    train.flatten()
    print(train.shape)

    test = np.array(test)
    testLabel = np.array(testLabel)
    test.flatten()
    print(test.shape)

    numberFilter = 20                                           #number of filter in each CNN Conv2D Layers
    kernelSize   = [3,5,7,9,11]                                 #sequentially size of kernel in layers 3*3 , 5*5
    poolSize     = 2
    padding = 'same'
    epoch = 10


    from sklearn.preprocessing import LabelBinarizer

    encoder = LabelBinarizer()
    trainLabel = encoder.fit_transform(trainLabel)               #transform string label to binary label
    print(trainLabel)
    testLabel = encoder.fit_transform(testLabel)
    print(testLabel)

    model = Sequential()                                                                                        #First Hidden layer
    print(train[0].shape)
    model.add(Conv2D(numberFilter,(kernelSize[0],kernelSize[0]),padding=padding,input_shape=train[0].shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(poolSize,poolSize)))

    model.add(Conv2D(numberFilter, (kernelSize[1], kernelSize[1]), padding=padding))                            #Second Hidden layer
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))

    model.add(Conv2D(numberFilter, (kernelSize[2], kernelSize[2]), padding=padding))                            #Third Hidden layer
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))

    model.add(Conv2D(numberFilter, (kernelSize[3], kernelSize[3]), padding=padding))                            #Fourth
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))

    """This 5th layer
    model.add(Conv2D(numberFilter, (kernelSize[4], kernelSize[4]), padding=padding))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    """

    model.add(Flatten())                                                                #n*m dimension data transform to k*1
    model.add(Dense(128))                                                               #reduce number of output to 128
    model.add(Dense(2))                                                                 #reduce number of output to 2
    model.add(Activation('softmax'))                                                    #decide to which output is target output
    model.compile(loss='sparse_categorical_crossentropy',optimizer="adadelta",metrics=['accuracy'])
    model.summary()                                                                     #show CNN layers and their size
    model.fit(train/255,trainLabel,epochs=epoch)                                        #input normalize to 0..1 and train CNN
    #model.predict(test)
    scores = model.evaluate(test,testLabel, verbose=1)                                  #give test data accuracy of success
    print(scores)
