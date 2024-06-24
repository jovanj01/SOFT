from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import cv2
from sklearn.metrics import mean_absolute_error
import csv
import matplotlib.patches as patches



folder_path = sys.argv[1]

pos_imgs = []
neg_imgs = []
pos_features = []
neg_features = []
labels = []

car_width = 500
car_height = 300

y_true = []
y_pred = []


# UCITAVANJE I PRIKAZ SLIKA

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')


#DETEKCIJA KONACNIH LINIJA CANNY metodom

def detect_line(img):
    # detekcija koordinata linije koristeci Hough transformaciju
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #trazim boju linije i nijanse oko nje
    central_color = np.array([158, 37, 37], dtype=np.uint8)
    color_range_lower = central_color - np.array([30, 30, 30], dtype=np.uint8)
    color_range_upper = central_color + np.array([30, 30, 30], dtype=np.uint8)
    #pravim masku za tu boju kako bih izdvojio samo liniju
    mask = cv2.inRange(img_RGB, color_range_lower, color_range_upper)

    gray_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    edges_img = cv2.Canny(gray_img, 50, 150, apertureSize=3)

    #plt.imshow(edges_img, "gray")

    # minimalna duzina linije
    min_line_length = 100
    
    # Hough transformacija
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi/180, threshold=50, lines=np.array([]), minLineLength=min_line_length, maxLineGap=200)
    
    #print("Detektovane linije [[x1 y1 x2 y2]]: \n", lines)
    
    x1 = lines[0][0][0]
    y1 = lines[0][0][1]
    x2 = lines[0][0][2]
    y2 = lines[0][0][3]
    plt.show()
    return (x1, y1, x2, y2)


#TRAZIMO KOODRINATE LINIJE
#OVO JE BESKORISNO
# def get_line_params(line_coords):
#     k = (float(line_coords[3]) - float(line_coords[1])) / (float(line_coords[2]) - float(line_coords[0]))
#     n = k * (float(-line_coords[0])) + float(line_coords[1])
#     return k, n

#METODA ZA DETEKCIJU PRELASKA PREKO LINIJE
def detect_cross(car_x, car_y, line_x1, line_y1, line_y2):
    return abs(car_x - line_x1) <= 100 and min(line_y1, line_y2) <= car_y <= max(line_y1, line_y2)

# RACUNA HOG I confidence score SVM klasifikatora
def classify_window(window):
    features = hog.compute(window).reshape(1, -1)
    return classifier.predict_proba(features)[0][1]

#procesiranje slike

def process_image(image, step_size, window_size=(car_width, car_height)):
    cars = {}
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            this_window = (y, x) # zbog formata rezultata
            window = image[y:y+window_size[1], x:x+window_size[0]]
            # plt.imshow(window, 'gray')
            # plt.show()
            window = cv2.resize(window, (120, 60), interpolation=cv2.INTER_NEAREST)
            # plt.imshow(window, 'gray')
            # plt.show()
            # if window.shape == (window_size[1], window_size[0]):
            if window.shape == (60, 120):
                score = classify_window(window)
                if score > 0.8:
                    cars[this_window] = score
    #hocu da removujem sve rectangleove koji nisu najbolji za jedan auto tako da za svaki auto ostane onaj sa najboljim skorom
    return get_best_rectangles(cars).keys()

def get_best_rectangles(rectangles):
    rectangles = sorted(rectangles.items(), key=lambda item: item[1], reverse=True)
    best_rectangles = dict()

    while len(rectangles) > 0:
        rectangle_temp = rectangles.pop(0)
        best_rectangle, rectangles = get_best_rectangle(rectangle_temp, rectangles)
        best_rectangles[best_rectangle[0]] = best_rectangle[1]

    return dict(best_rectangles)

def get_best_rectangle(current_rectangle, rectangles):
    temp = {}

    indexes = []

    temp[current_rectangle[0]] = current_rectangle[1]
    for i, rectangle in enumerate(rectangles):
        if does_overlap(current_rectangle[0], rectangle[0]):
            temp[rectangle[0]] = rectangle[1]
            indexes.append(i)

    best_rectangle = max(temp.items(), key=lambda x: x[1])
    remaining_rectangles = [rectangles[i] for i, _ in enumerate(rectangles) if i not in indexes]

    return best_rectangle, remaining_rectangles


def does_overlap(rectangle1, rectangle2):
    x1, y1 = rectangle1
    x2, y2 = rectangle2
    width, height = car_width, car_height
    return abs(y1 - y2) <= height and abs(x1 - x2) <= width

#PROCESIRANJE VIDEA
def process_video(video_path):
    collisions_counter = 0
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num) # indeksiranje frejmova
    
    # analiza videa frejm po frejm
    while True:
        frame_num += 1
        grabbed, frame = cap.read()
        #print(frame_num)

        # ako frejm nije zahvacen
        if not grabbed:
            break
        
        if frame_num % 10 == 1: # ako je prvi frejm, detektuj liniju
            line_coords = detect_line(frame)
            #print(line_coords)
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = process_image(frame_gray, step_size=100)
        plt.imshow(frame_gray, "gray")

        # Ispis rectangleova oko auta

        for car in cars:
            rectangle = patches.Rectangle((car[1], car[0]), car_width, car_height, linewidth=1, edgecolor='r', facecolor='none')
            ax = plt.gca()
            ax.add_patch(rectangle)

        # Show the plot
        plt.show()
        
        for car in cars:
            starting_y, starting_x = car
            #uzimamo sredinu auta po visini i duzini jer vecina auta prelazi preko spica auta, a svaki prelazi preko sredine
            car_width_middle = starting_x + car_width / 2
            car_height_middle = starting_y + car_height / 2
            
            if detect_cross(car_width_middle, car_height_middle, line_coords[0], line_coords[1], line_coords[3]):
                collisions_counter += 1
                print(collisions_counter)
    cap.release()
    return collisions_counter

#POZIVANJE METODA:

for img_name in os.listdir('data2/pictures/'):
    img_path = os.path.join('data2/pictures/', img_name)
    img = load_image(img_path)
    if 'p_' in img_name:
        pos_imgs.append(img)
    elif 'n_' in img_name:
        neg_imgs.append(img)

# print("Positive images #: ", len(pos_imgs))
# print("Negative images #: ", len(neg_imgs))

# Racunanje HOG deskriptora za slike iz MNIST skupa podataka
img_size = (60, 120)
nbins = 9
cell_size = (8, 8)
block_size = (2, 2)
hog = cv2.HOGDescriptor(_winSize=(img_size[1] // cell_size[1] * cell_size[1],
                                    img_size[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

for img in pos_imgs:
    pos_features.append(hog.compute(img))
    labels.append(1)

for img in neg_imgs:
    neg_features.append(hog.compute(img))
    labels.append(0)

pos_features = np.array(pos_features)
neg_features = np.array(neg_features)
x = np.vstack((pos_features, neg_features))
y = np.array(labels)

#obucavanje SVM
classifier = SVC(kernel='linear', probability=True) 
classifier.fit(x, y)
y_train_pred = classifier.predict(x)
# print("Train accuracy: ", accuracy_score(y, y_train_pred))

#ISPIS
file = open(f'{folder_path}counts.csv')
reader = csv.reader(file)

header = next(reader)

for row in reader:
    value = process_video(f'{folder_path}videos/{row[0]}.mp4')
    print(f'{row[0]}.mp4-{row[1]}-{value}')
    y_true.append(int(row[1]))
    y_pred.append(value)

print(f'MAE: {mean_absolute_error(y_true, y_pred)}')

# plt.show()