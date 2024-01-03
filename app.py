import numpy as np
import pickle
from flask import Flask,render_template,request
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from keras.utils import pad_sequences
import string
from pickle import dump
from pickle import load
from keras.applications.xception import Xception #to get pre-trained model Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.text import Tokenizer #for text tokenization
from keras.utils import to_categorical
#from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense #Keras to build our CNN and LSTM
from keras.layers import LSTM, Embedding, Dropout
from tqdm.notebook import tqdm as tqdm #to check loop progress
tqdm().pandas()

app=Flask(__name__,template_folder='template')
model = pickle.load(open("model.pkl", "rb"))
@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile=request.files['imagefile']
    image_path="./image/"+imagefile.filename
    imagefile.save(image_path)
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import argparse
    def extract_features(filename, model):
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4:
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature
    def word_for_id(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None
    def generate_desc(model, tokenizer, photo, max_length):
        in_text = 'start'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            pred = model.predict([photo,sequence], verbose=0)
            pred = np.argmax(pred)
            word = word_for_id(pred, tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
        return in_text
    max_length = 32
    tokenizer = load(open("tokenizer.p","rb"))
    model = load_model('models/model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")

    photo = extract_features(image_path, xception_model)
    img = Image.open(image_path)
    description = generate_desc(model, tokenizer, photo, max_length)
    msg=description[5:-4]
    from translate import Translator
    translator= Translator(to_lang="te")
    translation = translator.translate(msg)
    return render_template('index.html',prediction=description[5:-4],prediction1=translation)
    
if __name__=='__main__':
    app.run(port=3000,debug=True)