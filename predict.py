import numpy as np
import os
from scipy import misc
from keras.models import model_from_json
import pickle

classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()

# load json and create model
json_file = open("model/model_face.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model/model_face.h5")

for img in os.listdir("predict"):
    if img.startswith("."):
        continue

    image = np.array(misc.imread("predict/" + img))
    image = misc.imresize(image, (64, 64))
    image = np.array([image])
    image = image.astype("float32")
    image = image / 255.0

    prediction = loaded_model.predict(image)
    classification = int_to_word_out[np.argmax(prediction)]

    print(prediction)
    print(np.max(prediction))
    print(classification)

    folder = "output/" + classification
    if not os.path.exists(folder):
        os.makedirs(folder)

    os.rename("predict/" + img, folder + "/" + img)
