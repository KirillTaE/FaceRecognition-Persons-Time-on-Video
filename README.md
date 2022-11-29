# Отслеживание, распознавание лиц, а также подсчет времени присутствия на видео при помощи facenet_pytorch (MTCNN, InceptionResnetV1) и OpenCV

## Начало работы

1. Клонирование репозитория:
    
    ```bash
    git clone https://github.com/KirillTaE/facenet_pytorch.git
 
1. Установка необходимых библиотек:

    ```bash
    #facenet-pytorch==2.5.2
    #opencv-python==4.6.0.66
    #pandas==1.5.1
    #matplotlib==3.6.2
    pip install -r requirements.txt

1. Запуск программы с параметрами по умолчанию:

    ```bash
    python main.py

Передаваемые параметры:

  * `-pcp` или `--photo_catalog_path` Путь до каталога с фотографиями. Внутри данного каталога должны находиться подписанные именами папки с фотографиями людей, которые будут присутствовать на видео (default=`photos`)
  * `-ud` или `--update_data` Обновляет/Создает файл data.pt (вызывать при обновлении фотографий). В этом файле лежат закодированные лица людей (эмбеддинги) (default=`True`)
  * `-dp` или `--data_path` Путь до каталога, где находится файл data.pt (default=` ` - папка где находится программа)
  * `-wc` или `--web_cam` Считывать видео с веб-камеры (default=`False`)
  * `-ivp` или `--input_video_path` Полный путь входного видео (default=`video.mp4`)
  * `-dv` или `--do_video` True - создать видео, в котором будут отмечены все найденные и распознанные лица. False - не создавать (default=`False`)
  * `-ovp` или `--output_video_path` Полный путь выходного видео с отмеченными лицами (default=`new_video.mp4`)

Пример вызова программы с параметрами: `python main.py -pcp photos -ud True -ivp video.mp4 -dv True -ovp new_video.mp4`






