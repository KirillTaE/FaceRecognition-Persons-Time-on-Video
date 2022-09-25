from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2 as cv

data_path = 'data.pt'

def face_match(img_path):  # img_path= location of photo, data_path= location of data.pt
    # getting embedding matrix of the given img
    #global resnet
    #img = Image.open('b.jpg')
    img = Image.fromarray(img_path) #without read file
    face, prob = mtcnn(img, return_prob=True)  # returns cropped face and probability
    if prob == None:
        return ('no face', '?')
    emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false

    saved_data = torch.load(data_path)  # loading data.pt file
    embedding_list = saved_data[0]  # getting embedding data
    name_list = saved_data[1]  # getting list of names
    dist_list = []  # list of matched distances, minimum distance is used to identify the person

    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)
    min_dist_list = min(dist_list)
    if min_dist_list>1.1:
        return ('unknown', '?')
    idx_min = dist_list.index(min_dist_list)
    return (name_list[idx_min], min_dist_list)

def collate_fn(x):
    return x[0]

def update_data():
    dataset = datasets.ImageFolder('photos')  # photos folder path
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}  # accessing names of peoples from folder names

    loader = DataLoader(dataset, collate_fn=collate_fn)

    #face_list = []  # list of cropped faces from photos folder
    name_list = []  # list of names corrospoing to cropped photos
    embedding_list = []  # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet
    #global resnet
    for img, idx in loader:
        face, prob = mtcnn(img, return_prob=True)
        if face is not None and prob > 0.90:  # if face detected and porbability > 90%
            emb = resnet(face.unsqueeze(0))  # passing cropped face into resnet model to get embedding matrix
            embedding_list.append(emb.detach())  # resulten embedding matrix is stored in a list
            name_list.append(idx_to_class[idx])  # names are stored in a list

    data = [embedding_list, name_list]
    torch.save(data, data_path)  # saving data.pt file

mtcnn = MTCNN(margin=0, min_face_size=20)  # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval()  # initializing resnet for face img to embeding conversion
#resnet.classify = True

cap = cv.VideoCapture(0)
ret, im = cap.read()
y, x, _ = im.shape
y, x = y//2, x//2
y1, y2, x1, x2 = y - 180, y + 180, x - 130, x + 130
'''
________________________________
'''
update_data()
'''
________________________________
'''

#import time
#start_time = time.time()
while True:
    ret, image = cap.read()
    image_crop = image[y1:y2, x1:x2, :]

    result = face_match(image_crop)

    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #так даже дольше получилось, ну или я криво измерил лол
    cv.putText(image, f'Face matched with: {result[0]} With distance: {result[1]}',(30,30),cv.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
    cv.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv.imshow('me', image)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    if key == ord(' '):
        cv.waitKey()
#print("--- %s seconds ---" % (time.time() - start_time))


cv.destroyWindow()
