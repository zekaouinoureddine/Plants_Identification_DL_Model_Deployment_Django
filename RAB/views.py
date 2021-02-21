from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
from django.conf.urls import url

# Create your views here.

model = load_model('PId_Best.h5')
labels = ['daisy','dandelion','rose', 'sunflower', 'tulip']
img_heigh, img_with = 150, 150

def index(request):
    context = {}
    context["url"] = '/static/images/examp.jpg'
    img_path = '.'+context["url"]
    img = load_img(img_path, target_size = (img_heigh, img_with))
    x = img_to_array(img)
    x = x/255
    x = x.reshape(1, img_heigh, img_with, 3)
    pred = model.predict(x)
    
    import numpy as np
    context['predictedClass'] = labels[np.argmax(pred[0])]
    context['probability']    = "{:.2f}".format(round(np.max(pred), 2)*100)
    return render(request,'index.html',context)

def predImg(request):

    if request.method == 'POST':
        
        context = {}
        uploaded_file= request.FILES['img']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)

        context["url"] = fs.url(name)
        print(context["url"])
        testimage = '.'+context["url"]
        img = image.load_img(testimage, target_size=(img_heigh, img_with))
        x = image.img_to_array(img)

        x = x/255
        x = x.reshape(1, img_heigh, img_with, 3)
        pred = model.predict(x)

        import numpy as np
        context['predictedClass'] = labels[np.argmax(pred[0])]
        context['probability']    = "{:.2f}".format(round(np.max(pred), 2)*100)

    return render(request,'index.html',context)
