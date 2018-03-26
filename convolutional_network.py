import numpy as np
import scipy.misc
from IPython.core.display import SVG

from keras.models import Sequential
from keras.layers import Dense

from keras.utils.visualize_util import plot
from keras.utils.visualize_util import model_to_dot

import matplotlib.pyplot as plt

"""
    Reads image file and converts it to vector.
    Normalizes the vector before returning.
    @author gautamshetty
"""
def read_one_image_and_convert_to_vector(file_name):

    img = scipy.misc.imread(file_name).astype(np.float32)  # read image and convert to float
    img = img.ravel()

    """
        Normalize - subtract mean from all values.
    """
    mean_value = np.mean(img)
    img = img - mean_value

    std_dev = np.std(img)

    img /= std_dev

    return img

"""
    Creates input samples by reading image files and converting them to vectors.
"""
def create_samples(fileDirectory, num_of_samples=100):

    i_count = num_of_samples/100
    k_count = 10

    file_list = []
    for l in (range(num_of_samples/(i_count * k_count))):
        j = l * num_of_samples/100
        for k in range(10):
            for i in range(i_count):
                #print str(k) + "_" + str(j) + ".png"
                imgVector = read_one_image_and_convert_to_vector(fileDirectory + "/" + str(k) + "_" + str(j) + ".png")
                file_list.append(imgVector)
                j += 1
            j -= i_count

    return np.array(file_list)

def create_test_data(fileDirectory, num_of_samples=100):

    file_list = []

    for i in range(num_of_samples):
        imgVector = read_one_image_and_convert_to_vector(fileDirectory + "/" + str(i) + ".png")
        file_list.append(imgVector)

    return np.array(file_list)

input_data = []

input_data = create_samples("train", 2000)

test_data = create_samples("set3_100", 100)

model = Sequential()

model.add(Dense(100, input_dim=784, init='uniform', activation='relu'))
model.add(Dense(784, input_dim=100, init='uniform', activation='linear'))

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(input_data, input_data, nb_epoch=100, batch_size=10)

scores = model.evaluate(test_data, test_data, batch_size=1)

array_784 = model.predict(test_data)

trained_weights = np.array(model.get_weights())

tw_T = trained_weights[0].T

for i in range(tw_T.shape[0]):

    image_data = tw_T[i].reshape(28, 28)
    scipy.misc.imsave("task4/" + str(i) + ".png", image_data)

print trained_weights