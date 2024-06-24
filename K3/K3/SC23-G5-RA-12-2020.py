import numpy as np
import cv2 # OpenCV
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = 12,8
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
import os
import sys
import csv
from sklearn.cluster import KMeans

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def dilate(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32) # dati ulaz
    y_train = np.array(y_train, np.float32) # zeljeni izlazi na date ulaze
    
    print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    result = []
    result_temp = ''
    for output in outputs:
        result.append(alphabet[winner(output)])

    for res in result:
        result_temp = result_temp + res

    return result_temp

def concatenate_and_remove_duplicates(str1, str2):
    combined_str = str1 + str2
    unique_chars = ''
    
    for char in combined_str:
        if char not in unique_chars:
            unique_chars += char
    
    return unique_chars

def is_white_above(image_bin, x, y, w, h, margin=30):
    # Proveri da li je gornja ivica regiona u opsegu od 20 piksela potpuno bela
    above_roi = image_bin[max(0, y - margin):y, x:x + w]
    return np.any(above_roi > 200)

def select_roi(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po X osi
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 100 and h < 80 and h > 35 and y < 270 and x > 245 and x < 845:            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
           
            if is_white_above(image_bin, x, y, w, h, margin=30):
                # print("Skontao je kvacicu/tacku")
                y -= 20
                h += 20
           
            # oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions

def select_roi_with_distances(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po X osi
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if area > 100 and h < 80 and h > 35 and y < 270 and x > 245 and x < 845:            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
           
            if is_white_above(image_bin, x, y, w, h, margin=30):
                # print("Skontao je kvacicu/tacku")
                y -= 20
                h += 20
           
            # oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # izdvojiti sortirane parametre opisujucih pravougaonika
    # izracunati rastojanja izmedju svih susednih regiona po X osi i dodati ih u niz rastojanja
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2]) # x_next - (x_current + w_current)
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances

def display_result_with_spaces(outputs, alphabet, k_means):
    # odredjivanje indeksa grupe koja odgovara rastojanju izmedju reci
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    # iterativno dodavanje prepoznatih elemenata
    # dodavanje space karaktera ako je rastojanje izmedju dva slova odgovara rastojanju izmedju reci
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result

def hamming_distance(str1, str2):
    # Provera da li su stringovi iste dužine
    distance = 0

    if len(str1) != len(str2):
        distance = abs(len(str1) - len(str2))

    # Računanje Hemingovog rastojanja
    distance += sum(c1 != c2 for c1, c2 in zip(str1, str2))
    
    return distance

#POZIVANJE F-JA

folder_path = sys.argv[1]

file = open(f'{folder_path}res.csv', errors='ignore', encoding='utf-8')
reader = csv.reader(file)

header = next(reader)

unique_input = []
unique_letters = ''
# temp = 'kleptojšvsazmědcháníčťýiréžybú' niz dobijem ispisom konkateniranih reci uz eliminaciju duplikata slaaay
important_index_list = [ 0, 1, 2, 3, 9, 10, 13, 15, 16, 18, 21, 24, 26, 27, 28, 32, 33, 35, 38, 39, 42, 46, 59, 63, 69, 70, 72, 76, 83, 88 ]


for row in reader:   
     #ucitavanje slike i obrada...
    img_path = os.path.join(folder_path + f'/pictures/{row[0]}')
    img = load_image(img_path)
    img_gray = image_gray(img)
    img_bin = image_bin(img_gray)
    img_bin = erode(dilate(img_bin))
    img_inv = invert(img_bin)

    # display_image(img_inv)  
    new_img , regions = select_roi(img.copy(), img_inv)

    unique_input = unique_input + regions
    unique_letters = concatenate_and_remove_duplicates(unique_letters, row[1].replace(" ", ""))
    plt.show()

filtered_list = [unique_input[i] for i in important_index_list]
# print(len(filtered_list))

inputs = prepare_for_ann(filtered_list)
unique_letters_array = np.array(list(unique_letters))
outputs = convert_output(unique_letters_array)

ann = create_ann(output_size=len(inputs))
ann = train_ann(ann, inputs, outputs, epochs=2000)

file.close()

# Ponovno otvaranje fajla
file = open(f'{folder_path}res.csv', errors='ignore', encoding='utf-8')
reader = csv.reader(file)

# Preskakanje zaglavlja
header = next(reader)

# Ponovna iteracija kroz fajl
for row in reader:
    img_path = os.path.join(folder_path + f'/pictures/{row[0]}')
    img = load_image(img_path)
    img_gray = image_gray(img)
    img_bin = image_bin(img_gray)
    img_bin = erode(dilate(img_bin))
    img_inv = invert(img_bin)

    # display_image(img_inv)  
    new_img , regions, distances = select_roi_with_distances(img.copy(), img_inv)
    inputs_temp = prepare_for_ann(regions)
    result = ann.predict(np.array(inputs_temp, np.float32))

    if max(distances) > 10: #ovde gledam da je najveci razmak izmedju regiona veci od 10 i ako jeste smatramo ga razmakom (5 je premalo pa sam podizao malo po malo do 10)
        #kmeans 2 klastera 
        kmeans = KMeans(n_clusters=2, n_init='auto') #imam 2 
        kmeans.fit(np.array(distances).reshape(len(distances), 1))
        #ispis
        print(f'{row[0]} - {row[1]} - {display_result_with_spaces(result, unique_letters_array, kmeans)} - Rastojanje({hamming_distance(row[1], display_result_with_spaces(result, unique_letters_array, kmeans))})')
    else:
        #ispis
        print(f'{row[0]} - {row[1]} - {display_result(result, unique_letters_array)} - Rastojanje({hamming_distance(row[1], display_result(result, unique_letters_array))})')


    # display_image(new_img)
    plt.show()

plt.show()