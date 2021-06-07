from tensorflow.keras.models import load_model
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np



np.set_printoptions(linewidth = 200)
image_path = "C:/Users/Constantin/Documents/Ben's shit/GitHub/Handrwitten-Text-Recognition-System/My Handwriting images/processed images"

models_to_load_paths = ["C:/Users/Constantin/Documents/Ben's shit/GitHub/Handrwitten-Text-Recognition-System/Models/Model from Trial 4",
                  "C:/Users/Constantin/Documents/Ben's shit/GitHub/Handrwitten-Text-Recognition-System/Models/Model from Trial 5"]
models_to_load_names = ["Trial 4 Model", "Trial 5 model"]

correct_characters = ['D  ','C/c','A  ', 'a  ', 't  ', 'O/o', 'n  ', 'h  ', 'M/m', 'S/s', 'a  ', 'b  ',
                      'g  ', 'n  ', 'M/m', 'q  ', 'r  ', 'Z/z', 'Y/y', 'S/s', 'h  ', 'T  ', 'f  ', 'Q  ',
                      'A  ', 'B  ', 'N  ', 'M/m', 'L/l', 'C/c', 'C/c']

print("")
def zero():
    return '0'


def one():
    return '1'


def two():
    return '2'


def three():
    return '3'


def four():
    return '4'


def five():
    return '5'


def six():
    return '6'


def seven():
    return '7'


def eight():
    return '8'


def nine():
    return '9'


def A():
    return 'A'


def B():
    return 'B'


def C_or_c():
    return 'C/c'


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


def I_or_i():
    return 'I/i'


def J_or_j():
    return 'J/j'


def K_or_k():
    return 'K/k'


def L_or_l():
    return 'L/l'


def M_or_m():
    return 'M/m'


def N():
    return 'N'


def O_or_o():
    return 'O/o'


def P_or_p():
    return 'P/p'


def Q():
    return "Q"


def R():
    return 'R'


def S_or_s():
    return 'S/s'


def T():
    return 'T'


def U_or_u():
    return 'U/u'


def V_or_v():
    return 'V/v'


def W_or_w():
    return 'W/w'


def X_or_x():
    return 'X/x'


def Y_or_y():
    return 'Y/y'


def Z_or_z():
    return 'Z/z'


def a():
    return 'a'


def b():
    return 'b'


def d():
    return 'd'


def e():
    return 'e'


def f():
    return 'f'


def g():
    return 'g'


def h():
    return 'h'


def n():
    return 'n'


def q():
    return 'q'


def r():
    return 'r'


def t():
    return 't'


def default():
    print("error")
    return


switcher = {
    0: zero(),
    1: one(),
    2: two(),
    3: three(),
    4: four(),
    5: five(),
    6: six(),
    7: seven(),
    8: eight(),
    9: nine(),
    10: A(),
    11: B(),
    12: C_or_c(),
    13: D(),
    14: E(),
    15: F(),
    16: G(),
    17: H(),
    18: I_or_i(),
    19: J_or_j(),
    20: K_or_k(),
    21: L_or_l(),
    22: M_or_m(),
    23: N(),
    24: O_or_o(),
    25: P_or_p(),
    26: Q(),
    27: R(),
    28: S_or_s(),
    29: T(),
    30: U_or_u(),
    31: V_or_v(),
    32: W_or_w(),
    33: X_or_x(),
    34: Y_or_y(),
    35: Z_or_z(),
    36: a(),
    37: b(),
    38: d(),
    39: e(),
    40: f(),
    41: g(),
    42: h(),
    43: n(),
    44: q(),
    45: r(),
    46: t()
}


def convert_to_classes(prediction_output):
    return switcher.get(prediction_output,default)

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



