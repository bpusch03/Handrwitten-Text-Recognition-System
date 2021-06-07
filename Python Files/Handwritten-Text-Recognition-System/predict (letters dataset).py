from tensorflow.keras.models import load_model
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np



np.set_printoptions(linewidth = 200)
image_path = "C:/Users/Constantin/Documents/Ben's shit/GitHub/Handrwitten-Text-Recognition-System/My Handwriting images/processed images"

models_to_load_paths = ["C:/Users/Constantin/Documents/Ben's shit/GitHub/Handrwitten-Text-Recognition-System/Models/Model from Trial 4 with letters dataset",
                  "C:/Users/Constantin/Documents/Ben's shit/GitHub/Handrwitten-Text-Recognition-System/Models/Model from Trial 5 with letters dataset"]
models_to_load_names = ["Trial 4 Model with letters dataset", "Trial 5 model with letters data set"]

correct_characters = ['D','C','A', 'a', 't', 'O', 'n', 'h', 'M', 'S', 'a', 'b',
                      'g', 'n', 'M', 'q', 'r', 'Z', 'Y', 'S', 'h', 'T', 'f', 'Q',
                      'A', 'B', 'N', 'M', 'L', 'C', 'C']

print("")


def A():
    return 'A'


def B():
    return 'B'


def C():
    return 'C'


def D():
    return 'D'


def E():
    return 'E'


def F():
    return 'F'


def G():
    return 'G'


def H():
    return 'H'


def I():
    return 'I'


def J():
    return 'J'


def K():
    return 'K'


def L():
    return 'L'


def M():
    return 'M'


def N():
    return 'N'


def O():
    return 'O'


def P():
    return 'P'


def Q():
    return "Q"


def R():
    return 'R'


def S():
    return 'S'


def T():
    return 'T'


def U():
    return 'U'


def V():
    return 'V'


def W():
    return 'W'


def X():
    return 'X'


def Y():
    return 'Y'


def Z():
    return 'Z'


switcher = {
    1: A(),
    2: B(),
    3: C(),
    4: D(),
    5: E(),
    6: F(),
    7: G(),
    8: H(),
    9: I(),
    10: J(),
    11: K(),
    12: L(),
    13: M(),
    14: N(),
    15: O(),
    16: P(),
    17: Q(),
    18: R(),
    19: S(),
    20: T(),
    21: U(),
    22: V(),
    23: W(),
    24: X(),
    25: Y(),
    26: Z(),

}


def convert_to_classes(prediction_output):
    return switcher.get(prediction_output)

def predict(images,model,model_name):
    print("")
    print("")
    print(model_name + ':')
    print("Actual Character             Model's Prediction")
    prediction = np.argmax(model.predict(images), axis = 1)
    for i in range(0,len(images)):
        print(correct_characters[i] + "                                            " + convert_to_classes(prediction[i]))




images = []

for i in range(1,32):
    images.append(np.array(Image.open(image_path + '/' + str(i) + '.jpg')))

images = np.array(images)
images = images.reshape(images.shape[0],28,28,1)
images = images.astype('float32')
images = images / 255.0

models = []
for paths in models_to_load_paths:
    models.append(load_model(paths,compile=True))

for m in range(0, len(models)):
    predict(images,models[m],models_to_load_names[m])



