import os
import cv2
import sys
import time
import pickle
import simplejson
import collections
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
from config import Config

from utils.io.frames_io import ImageDisplayer
from utils.io.plots import plot_confusion_matrix
from sklearn.metrics import classification_report
from utils.skeletons.skeleton_tracker import Tracker
from utils.neural_network.nn_tools import ClassifierTrain
from utils.skeletons.openpose_utils import SkeletonDetector
from utils.io.filemanager import save_list, get_filenames,read_list
from utils.skeletons.feature_processing import extract_multi_frame_features
from utils.io.skeletons_io import ReadValidImagesAndActionTypesByTxt, load_skeleton_data

def convert_video_to_images():
    dataset_path = os.path.join('.', 'input_dataset', '')
    dataset_folders = [os.path.join(dataset_path, name) for name
                       in os.listdir(dataset_path) if
                       os.path.isdir(os.path.join(dataset_path, name))]

    for folder_path in dataset_folders:
        files_in_folder = os.listdir(folder_path)
        class_name = folder_path.split(os.sep)[-1].replace('_', '')

        if len(files_in_folder):
            print(f'Обходим папку {folder_path}, в которой {len(files_in_folder)} файлов. Класс: {class_name}')

        for file in files_in_folder:
            filename_without_format = file.split('.')[0]
            path_to_save = os.path.join('.', 'data', f'{class_name}_{filename_without_format}')
            os.makedirs(path_to_save, exist_ok=True)

            video_capture = cv2.VideoCapture(os.path.join(folder_path, file))
            success, image = video_capture.read()
            count = 0

            while success:
                cv2.imwrite(os.path.join(path_to_save, "{:05d}".format(count) + ".jpg"), image)
                success, image = video_capture.read()
                print(f'Класс: {class_name}, Видео: {file}, Кадр: {count}')
                count += 1


def prepare_images(data_images_file):
    data_processed_path = os.path.join('data_processed', 'raw_skeletons')
    skeleton_detector = SkeletonDetector(Config.OPENPOSE_MODEL, Config.IMAGE_SIZE)
    people_tracker = Tracker()
    images_loader = ReadValidImagesAndActionTypesByTxt(
        img_folder=os.path.join('.', 'data'),
        valid_imgs_txt=data_images_file,
        img_filename_format=Config.IMAGE_FORMAT
    )

    # Сохранение информации о изображениях
    images_loader.save_images_info(filepath=os.path.join(data_processed_path, 'images_info.json'))

    img_displayer = ImageDisplayer("Detected skeleton")
    image_visual_path = os.path.join(data_processed_path, 'image_visual')
    skeleton_result_path = os.path.join(data_processed_path, 'skeleton_result')

    # Создаем выходные папки
    os.makedirs(skeleton_result_path, exist_ok=True)
    os.makedirs(image_visual_path, exist_ok=True)

    # Считываем изображения и обрабатываем
    num_total_images = images_loader.num_images
    for ith_img in range(num_total_images):
        img, str_action_label, img_info = images_loader.read_image()

        # Детектируем людей
        humans = skeleton_detector.detect(img)

        # Рисуем скелет
        img_disp = img.copy()
        skeleton_detector.draw(img_disp, humans)
        img_displayer.show_frame(img_disp)

        # Информация о скелете сохраняется в файл
        skeletons, scale_h = skeleton_detector.people_to_skeletons(humans)
        dict_id2skeleton = people_tracker.track(skeletons)  # dict: (int human id) -> (np.array() skeleton)

        # Подготовка данных скелета для сохранения
        skels_to_save = [img_info + skeleton.tolist() for skeleton in dict_id2skeleton.values()]

        # Сохранение результата
        save_list(os.path.join(skeleton_result_path, Config.SKELETON_FORMAT.format(ith_img)), skels_to_save)
        cv2.imwrite(os.path.join(image_visual_path, Config.IMAGE_FORMAT.format(ith_img)), img_disp)

        print(f"{ith_img}/{num_total_images} | {len(skeletons)} людей найдено")


def get_length_of_one_skeleton_data(file_paths):
    data_processed_path = os.path.join('data_processed', 'raw_skeletons', 'skeleton_result')

    for i in range(len(file_paths)):
        skeletons = read_list(os.path.join(data_processed_path, Config.SKELETON_FORMAT.format(i)))
        if len(skeletons):
            data_size = len(skeletons[0])
            assert (data_size == 41)
            return data_size

    raise RuntimeError(f"No valid txt under: {file_paths}.")


def create_skeleton_files():
    data_skeleton_processed_path = os.path.join('data_processed', 'raw_skeletons', 'skeleton_result')
    data_processed_path = os.path.join('data_processed', 'raw_skeletons')
    file_paths = get_filenames(data_skeleton_processed_path, use_sort=True, with_folder_path=True)
    num_skeletons = len(file_paths)
    data_length = get_length_of_one_skeleton_data(file_paths)

    print(f'Data length of one skeleton is {data_length}')

    # -- Read in skeletons and push to all_skeletons
    all_skeletons = []
    labels_cnt = collections.defaultdict(int)

    for i in range(num_skeletons):
        # Чтение скелетов из файла
        skeletons = read_list(os.path.join(data_skeleton_processed_path, Config.SKELETON_FORMAT.format(i)))

        if not skeletons:  # Если файл пустой, пропускаем это изображение
            continue

        skeleton = skeletons[0]
        label = skeleton[3]

        if label not in Config.CLASSES:  # Если метка неверная, пропускаем это изображение
            print(f'Неизвестный класс {label}')
            continue
        labels_cnt[label] += 1
        # Push to result
        all_skeletons.append(skeleton)

        # Print
        if i == 1 or i % 100 == 0:
            print("{}/{}".format(i, num_skeletons))

    # Сохранение данных
    with open(os.path.join(data_processed_path, 'images_info.json'), 'w') as f:
        simplejson.dump(all_skeletons, f)

    print(f"There are {len(all_skeletons)} skeleton data.")
    # print(f"They are saved to {DST_ALL_SKELETONS_TXT}")
    print("Number of each action: ")
    for label in Config.CLASSES:
        print(f" {label}: {labels_cnt[label]}")


def preprocess_features():
    data_processed_path = os.path.join('data_processed', 'raw_skeletons', 'images_info.json')
    X0, Y0, video_indices = load_skeleton_data(data_processed_path, Config.CLASSES)

    # Process features
    print("\nExtracting time-series features ...")
    X, Y = extract_multi_frame_features(X0, Y0, video_indices, Config.WINDOWS_SIZE, is_print=True)
    print(f"X.shape = {X.shape}, len(Y) = {len(Y)}")

    # Save data
    print("\nWriting features and labels to disk ...")
    features_X_path = os.path.join('data_processed', 'features_X.csv')
    features_Y_path = os.path.join('data_processed', 'features_Y.csv')
    np.savetxt(features_X_path, X, fmt="%.5f")
    print("Save features to: " + features_X_path)
    np.savetxt(features_Y_path, Y, fmt="%i")
    print("Save labels to: " + features_Y_path)


def train_test_split(X, Y, ratio_of_test_size):
    return sklearn.model_selection.train_test_split(X, Y, test_size=ratio_of_test_size, random_state=1)


""" Evaluate accuracy and time cost """


def evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y):
    # Accuracy
    t0 = time.time()
    tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
    print(f"Accuracy on training set is {tr_accu}")

    te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)
    print(f"Accuracy on testing set is {te_accu}")

    print("Accuracy report:")
    print(classification_report(te_Y, te_Y_predict, target_names=classes, output_dict=False))

    # Time cost
    average_time = (time.time() - t0) / (len(tr_Y) + len(te_Y))
    print("Time cost for predicting one sample: " + "{:.5f} seconds".format(average_time))

    # Plot accuracy
    axis, cf = plot_confusion_matrix(te_Y, te_Y_predict, classes)
    plt.show()


if __name__ == '__main__':
    welcome_messages = ['1. Разбить видео на изображения', '2. Обучение сети', '', 'Выберите режим работы: ']
    work_mode = int(input('\n'.join(welcome_messages)))

    if work_mode == 1:
        print('Разбивка видео на изображения...')
        convert_video_to_images()
        print('Изображения готовы')

    elif work_mode == 2:
        data_images_file = os.path.join('.', 'data', 'data_images.txt')

        if os.path.exists(data_images_file):
            # print('Готовим изображения...')
            # prepare_images(data_images_file)
            #
            # print('Обучение сети...')
            # create_skeleton_files()
            # print('Извлечение признаков...')
            # preprocess_features()

            # -- Загрузка обработанных данных
            print("\nСчитываем данные классов, особенностей и названий...")
            X = np.loadtxt(os.path.join('data_processed', 'features_X.csv'), dtype=float)  # features
            Y = np.loadtxt(os.path.join('data_processed', 'features_Y.csv'), dtype=int)  # labels

            tr_X, te_X, tr_Y, te_Y = train_test_split(X, Y, ratio_of_test_size=0.3)

            print("\nРазделение данных для обучения и тестирования:")
            print("Размер данных X: ", tr_X.shape)
            print("Количество данных для тренировки: ", len(tr_Y))
            print("Количество данных для тестирования: ", len(te_Y))

            print("\nОбучение модели...")
            model = ClassifierTrain()
            model.train(tr_X, tr_Y)

            print("\nОценка модели...")
            evaluate_model(model, Config.CLASSES, tr_X, tr_Y, te_X, te_Y)

            print(f"\nСохраняем обученную модель в {Config.SRC_MODEL_PATH}")
            with open(f'../{Config.SRC_MODEL_PATH}', 'wb') as f:
                pickle.dump(model, f)

        else:
            print(f'Ошибка! Разметка датасета не найдена... ({data_images_file})')
    else:
        print('Неизвестный режим работы')