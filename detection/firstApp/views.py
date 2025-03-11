from django.shortcuts import render

import os
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph
from tensorflow import Session
from PIL import Image

img_height, img_width=224,224
with open('./models/labelmap.json','r') as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model=tf.keras.models.load_model('models/Aug_Shell_mobileNet_v2.h5')

def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    try:
        fileObj=request.FILES['filePath']
        fs=FileSystemStorage()                                             
        filePathName=fs.save(fileObj.name,fileObj)
        filePathName=fs.url(filePathName)
        testimage='.'+filePathName
        img = image.load_img(testimage, target_size=(img_height, img_width))
        x = image.img_to_array(img)
        x=x/255
        x=x.reshape(1,img_height, img_width,3)
        with model_graph.as_default():
            with tf_session.as_default():
                predi=model.predict(x)

        import numpy as np
        predictedLabel=labelInfo[str(np.argmax(predi[0]))]

        acc = "{:.2f}".format(np.max(predi[0])*100)
        acc = float(acc)-6.00
        if float(acc)<93.50 or predictedLabel == labelInfo["2"]:
            predictedLabel = labelInfo["2"]
            acc=" "
        else:
            acc = "Classification Accuracy is "+str(acc)+"%"

        print(filePathName)
        context={'filePathName':filePathName,'predictedLabel':predictedLabel[0], 'acc':str(acc)}
        
        return render(request,'index.html',context) 
    except:
        return render(request,'errors.html')