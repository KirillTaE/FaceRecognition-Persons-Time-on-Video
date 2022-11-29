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

  * `-р` или `--help` Выводится описание всех передаваемых параметров
  * `-pcp` или `--photo_catalog_path` Путь до каталога с фотографиями. Внутри данного каталога должны находиться подписанные именами папки с фотографиями людей, которые будут присутствовать на видео (default=`photos`)
  * `-ud` или `--update_data` Обновляет/Создает файл data.pt (вызывать при обновлении фотографий). В этом файле лежат закодированные лица людей (эмбеддинги) (default=`True`)
  * `-dp` или `--data_path` Путь до каталога, где находится файл data.pt (default=` ` - папка где находится программа)
  * `-wc` или `--web_cam` Считывать видео с веб-камеры. Файлы с подсчетом времени создаваться не будут (default=`False`)
  * `-ivp` или `--input_video_path` Полный путь входного видео (default=`video.mp4`)
  * `-dv` или `--do_video` True - создать видео, в котором будут отмечены все найденные и распознанные лица. False - не создавать (default=`False`)
  * `-ovp` или `--output_video_path` Полный путь выходного видео с отмеченными лицами (default=`new_video.mp4`)

Пример вызова программы с параметрами: `python main.py -pcp photos -ud True -ivp video.mp4 -dv True -ovp new_video.mp4`

## Выходные данные

В конце работы программы будет создано 4 файла:
  * Time_{`--input_video_path`}.csv
  <img width="181" alt="image" src="https://user-images.githubusercontent.com/82940632/204598256-10575e3e-ad10-4419-a39f-6186298facfd.png">

  * Time_drop0_{`--input_video_path`}.csv 
  <img width="182" alt="image" src="https://user-images.githubusercontent.com/82940632/204598348-ec7662e5-0478-4a06-968f-aff6f9cb6b31.png">

  * barh_{`--input_video_path`}.pdf 
  <img width="825" alt="image" src="https://user-images.githubusercontent.com/82940632/204598040-2234cda0-edba-4a6a-8ced-2f7dc70eb2aa.png">
  
  * pie_{`--input_video_path`}.pdf 
  <img width="424" alt="image" src="https://user-images.githubusercontent.com/82940632/204597855-ad88af7a-226e-4487-9180-005455d15959.png">


