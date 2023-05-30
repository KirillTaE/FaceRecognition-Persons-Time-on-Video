from facenet_pytorch import MTCNN, InceptionResnetV1
from yolov5facedetector.face_detector import YoloDetector
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import time
import os


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pcp', '--photo_catalog_path', type=str, default='photos',
                        help="Путь до каталога с фотографиями. Внутри данного каталога должны находиться подписанные именами папки с фотографиями людей, которые будут присутствовать на видео (default='photos')")
    parser.add_argument('-ud', '--update_data', type=str, default="True",
                        help="Обновляет/Создает файл data.pt (вызывать при обновлении фотографий). В этом файле лежат закодированные лица людей (default=True)")
    parser.add_argument('-dp', '--data_path', type=str, default='',
                        help="Путь до каталога, где находится файл data.pt (default='')")
    parser.add_argument('-wc', '--web_cam', type=str, default="False",
                        help="Считывать видео с веб-камеры. Файлы с подсчетом времени создаваться не будут (default=False)")
    parser.add_argument('-ivp', '--input_video_path', type=str, default='zoom_1.mp4',
                        help="Полный путь входного видео (default='zoom_1.mp4)")
    parser.add_argument('-dv', '--do_video', type=str, default="False",
                        help="True - создать видео, где будут отмечены все найденные лица. False - не создавать (default=False)")
    parser.add_argument('-ovp', '--output_video_path', type=str, default='new_video.mp4',
                        help="Полный путь выходного видео с отмеченными лицами (default='new_video.mp4')")
    parser.add_argument('-m', '--model_name', type=str, default='mtcnn',
                        help="Модель для детектирования лиц (default='mtcnn')")

    return parser


parser = createParser()
namespace = parser.parse_args(sys.argv[1:])


# print(namespace)


def collate_fn(x):
    return x[0]


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def update_data(path):
    print("update_data")
    dataset = datasets.ImageFolder(path)  # photos folder path
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}  # accessing names of peoples from folder names

    loader = DataLoader(dataset, collate_fn=collate_fn)

    # face_list = []  # list of cropped faces from photos folder
    name_list = []  # list of names corrospoing to cropped photos
    embedding_list = []  # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

    if model_name == "mtcnn":
        mtcnn2 = MTCNN(margin=0, min_face_size=20, device=device)  # initializing mtcnn for face detection
        for img, idx in loader:
            face, prob = mtcnn2(img, return_prob=True)
            if face is not None and prob > 0.90:  # if face detected and porbability > 90%
                emb = resnet(face.unsqueeze(0))  # passing cropped face into resnet model to get embedding matrix
                embedding_list.append(emb.detach())  # resulten embedding matrix is stored in a list
                name_list.append(idx_to_class[idx])  # names are stored in a list

    if model_name == "yolo":
        Yolo = YoloDetector(target_size=720, gpu=0, min_face=50)
        for img, idx in loader:
            img = np.array(img)
            # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            boxes, prob, key_points = Yolo(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            box = boxes[0][0]
            face = img[box[1]:box[3], box[0]:box[2]]
            face = cv.resize(face, (160, 160), interpolation=cv.INTER_AREA).copy()
            if face is not None:  # and prob[0][0][0] > 0.90:  # if face detected and porbability > 90%
                image_tensor = torch.Tensor(face).permute(2, 0, 1)
                emb = resnet(fixed_image_standardization(image_tensor).unsqueeze(
                    0))  # passing cropped face into resnet model to get embedding matrix
                embedding_list.append(emb.detach())  # resulten embedding matrix is stored in a list
                name_list.append(idx_to_class[idx])  # names are stored in a list

    data = [embedding_list, name_list]
    torch.save(data, data_path)  # saving data.pt file


def face_match(img_path):  # img_path= location of photo, data_path= location of data.pt
    # getting embedding matrix of the given img
    # img = Image.open('b.jpg')
    # img = Image.fromarray(img_path)  # cv.cvtColor(img_path, cv.COLOR_BGR2RGB) #without read file
    img = img_path.copy()
    if model_name == "mtcnn":
        faces, prob = mtcnn(img, return_prob=True)  # returns cropped face and probability
        boxes, _ = mtcnn.detect(img)
        # print(faces[0])
    # print(prob)

    if model_name == "yolo":
        faces = []
        # img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        boxes, prob, key_points = Yolo(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        prob = prob[0]
        boxes = boxes[0]

        for box in boxes:
            face = img[box[1]:box[3], box[0]:box[2]]
            face = cv.resize(face, (160, 160), interpolation=cv.INTER_AREA).copy()
            image_tensor = torch.Tensor(face).permute(2, 0, 1)
            faces.append(fixed_image_standardization(image_tensor))
        # print(faces[0])

    if len(prob) == 0 or prob[0] is None:
        return {'no face': None}

    saved_data = torch.load(data_path)  # loading data.pt file
    embedding_list = saved_data[0]  # getting embedding data
    name_list = saved_data[1]  # getting list of names
    # list of matched distances, minimum distance is used to identify the person
    faces_list = []
    faces_dict = {}
    # print(len(faces))
    for index, face in enumerate(faces):
        dist_list = []
        emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false
        for emb_db in embedding_list:
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)

        min_dist_list = min(dist_list)
        idx_min = dist_list.index(min_dist_list)
        if model_name == "mtcnn":
            regularization = 1.1

            if min_dist_list < regularization:
                faces_dict[name_list[idx_min]] = (float('{:.6f}'.format(min_dist_list)), boxes[index])
            else:
                faces_dict[f'unknown_{index}'] = (float('{:.6f}'.format(min_dist_list)), boxes[index])

        if model_name == "yolo":
            regularization = 1.2

            if min_dist_list < regularization:
                # print(name_list[idx_min])
                faces_dict[name_list[idx_min]] = (float('{:.6f}'.format(min_dist_list)), boxes[index])
            else:
                faces_dict[f'unknown_{index}'] = (float('{:.6f}'.format(min_dist_list)), boxes[index])

    # print(len(faces_dict))
    return faces_dict


if __name__ == "__main__":
    torch.cuda.empty_cache()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('Running on device: {}'.format(device))
    # print(torch.cuda.device_count())
    ''''''
    # exit()

    PATH = "InceptionResnetV1_VGGFace2/InceptionResnetV1-vggface2.pt"
    try:
        resnet = torch.load(PATH)
        resnet.eval()
    except Exception:
        os.mkdir("InceptionResnetV1_VGGFace2")
        resnet = InceptionResnetV1(
            pretrained='vggface2').eval()  # initializing resnet for face img to embeding conversion
        torch.save(resnet, PATH)
    # exit()

    # resnet.classify = True

    doVideo = namespace.do_video.lower() == "true"
    updateData = namespace.update_data.lower() == "true"
    data_path = namespace.data_path + "data.pt"
    web_cam = namespace.web_cam.lower() == "true"
    model_name = namespace.model_name.lower()
    print(model_name)

    input_video_path = namespace.input_video_path
    input_photo_catalog_path = namespace.photo_catalog_path
    input_output_video_path = f"{model_name}_" + namespace.output_video_path
    ''''''''''''''''''
    if updateData:
        update_data(input_photo_catalog_path)
    ''''''''''''''''''
    if web_cam:
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(input_video_path)  # 0

    if model_name == "mtcnn":
        mtcnn = MTCNN(margin=0, min_face_size=40, keep_all=True, device=device)  # initializing mtcnn for face detection

    if model_name == "yolo":
        Yolo = YoloDetector(target_size=720, gpu=0, min_face=50)

    ret, im = cap.read()

    y, x, _ = im.shape

    scale_percent = 75  # percent of original size
    width = int(x * scale_percent / 100)
    height = int(y * scale_percent / 100)
    dim = (width, height)

    dy = y // 2
    dx = x // 2
    y, x = y // 2, x // 2
    y1, y2, x1, x2 = y - dy, y + dy, x - dx, x + dx

    fps = cap.get(cv.CAP_PROP_FPS)
    all_frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)

    nsec = 1  # interval between taking a new frame (sec)

    saved_data = torch.load(data_path)  # loading data.pt file
    names = {'no face': [0, True], 'unknown': [0, True]}
    for name in set(saved_data[1]):
        names[name] = [0, True]

    if (not web_cam) and doVideo:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_size = (frame_width, frame_height)
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v');
        writer = cv.VideoWriter(input_output_video_path, fourcc, 10, frame_size)
        # writer = cv.VideoWriter('drive/MyDrive/colab/celebr_zoom_new.mp4', fourcc, 10, frame_size)
        # font = ImageFont.truetype("drive/MyDrive/colab/arial.ttf", size=25)

    import copy

    numImage = 1
    alltime = 0
    N = -1
    eps = 0.001

    while True:
        ret, image = cap.read()
        if not ret:
            break
        # print(ret)

        numImage += 1
        if numImage % (fps * nsec) != 0:
            continue
        alltime += nsec
        image_crop = image[y1:y2, x1:x2, :]

        result = face_match(image_crop)

        # print(result)
        somebody = False
        if N == -1:
            for name, _ in result.items():
                if ("unknown_" in name):
                    if not somebody:
                        names["unknown"][0] += nsec
                        somebody = True
                        continue
                    else:
                        continue
                names[name][0] += nsec
            N = 0
        else:
            for name, _ in result.items():
                if ("unknown_" in name):
                    if not somebody:
                        names["unknown"][0] += nsec
                        somebody = True
                        continue
                    else:
                        continue
                if result_3.get(name):
                    if (abs(result[name][0] - result_3[name][0]) >= eps):
                        names[name][0] += nsec
                        names[name][1] = True
                    else:
                        names[name][1] = False
                else:
                    names[name][0] += nsec
                    names[name][1] = True

        result_3 = copy.deepcopy(result)

        if web_cam or doVideo:
            # frame_draw = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB)).copy()
            # draw = ImageDraw.Draw(frame_draw)
            for name, box in result.items():
                if box is not None:
                    # draw.rectangle(box[1].tolist(), outline=(255, 0, 0), width=6)
                    cv.rectangle(image, (int(box[1][0]), int(box[1][1])), (int(box[1][2]), int(box[1][3])), (0, 0, 255),
                                 2)
                    if ("unknown_" in name) or names.get(name)[1] == True:
                        # draw.text((box[1][0],box[1][1]), name, font=font,) #fill="black" fill="blue"
                        cv.putText(image, name, (int(box[1][0] + 5), int(box[1][1] + 10)), cv.FONT_HERSHEY_COMPLEX, 0.5,
                                   (255, 255, 255), 1)
                    else:
                        # draw.text((box[1][0],box[1][1]), name + "\n(Sleeping)", font=font,) #fill="black" fill="blue"
                        cv.putText(image, name + "(Sleeping)", (int(box[1][0] + 5), int(box[1][1] + 10)),
                                   cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            # Записываем фрейм в выходные файлы
            # writer.write(cv.cvtColor(np.array(frame_draw), cv.COLOR_RGB2BGR))
            if web_cam:
                cv.imshow('frame', cv.resize(image, dim))
                if cv.waitKey(100) == 27:  # Клавиша Esc
                    cv.destroyAllWindows()
                    exit()

            if doVideo:
                writer.write(image)
        if not web_cam:
            print(f"Done {int(numImage * 100 / all_frame_count)}%")
    print(f"Done !!!")
    cv.destroyAllWindows()
    if doVideo:
        writer.release()

    new_names = {key: int(val[0]) for key, val in sorted(names.items(), key=lambda item: item[1][0], reverse=True)}
    new_names_drop0 = {key: int(val[0]) for key, val in sorted(names.items(), key=lambda item: item[1][0], reverse=True)
                       if val[0] != 0}

    new_names_percent = {key: str(val) + " seconds: " + str(int(val * 100 // alltime)) + "%" for key, val in
                         new_names.items()}
    new_names_percent_drop0 = {key: str(val) + " seconds: " + str(int(val * 100 // alltime)) + "%" for key, val in
                               new_names.items() if val != 0}

    new_names_df = pd.DataFrame(list(new_names.items()), columns=['Names', 'Time'])
    new_names_drop0_df = pd.DataFrame(list(new_names_drop0.items()), columns=['Names', 'Time'])

    new_names_df["Time %"] = new_names_df["Time"] * 100 // alltime
    new_names_drop0_df["Time %"] = new_names_drop0_df["Time"] * 100 // alltime

    #
    # new_names_df = pd.read_csv("new_names_df.csv").reset_index()
    # new_names_drop0_df = pd.read_csv("new_names_drop0_df.csv").reset_index()
    ######Output########
    new_names_df[::-1].plot(kind="barh", x="Names", y="Time %", figsize=(20, 5), xticks=range(0, 101, 5))
    plt.savefig(f'{model_name}_barh_{input_video_path}.jpg')
    new_names_drop0_df.groupby(['Names']).sum().plot(kind='pie', y='Time', autopct='%1.0f%%', figsize=(10, 10),
                                                     legend=False)
    plt.savefig(f'{model_name}_pie_{input_video_path}.jpg')

    ###############

    new_names_norm = {key: time.strftime("%H:%M:%S", time.gmtime(int(val[0]))) for key, val in
                      sorted(names.items(), key=lambda item: item[1][0], reverse=True)}
    new_names_drop0_norm = {key: time.strftime("%H:%M:%S", time.gmtime(int(val[0]))) for key, val in
                            sorted(names.items(), key=lambda item: item[1][0], reverse=True)
                            if val[0] != 0}

    new_names_df_norm = pd.DataFrame(list(new_names_norm.items()), columns=['Names', 'Time'])
    new_names_drop0_df_norm = pd.DataFrame(list(new_names_drop0_norm.items()), columns=['Names', 'Time'])

    new_names_df_norm["Time %"] = new_names_df["Time"] * 100 // alltime
    new_names_drop0_df_norm["Time %"] = new_names_drop0_df["Time"] * 100 // alltime

    new_names_df_norm.to_csv(f"{model_name}_Time_{input_video_path}.csv", index=False, sep=';')
    new_names_drop0_df_norm.to_csv(f"{model_name}_Time_drop0_{input_video_path}.csv", index=False, sep=';')

    ###############
