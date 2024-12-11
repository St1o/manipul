import os
import sys

import cv2
import time
import numpy as np

import utils.io.plots as plots
import utils.io.frames_io as lib_images_io
from config import Config
from utils.io.filemanager import save_list
from utils.io.network_io import NetworkManager
from utils.skeletons.skeleton_tracker import Tracker
from utils.skeletons.openpose_utils import SkeletonDetector
from utils.neural_network.person_classifier import PersonClassifier
from utils.utils import remove_bad_skeletons, get_video_reader, save_skeleton_data

def draw_result_frame(frame, index_frame, man, dict_id2skeleton,skeleton_detector, person_classifier, desired_rows, dict_id2label, scale_h):
    r, c = frame.shape[0:2]
    desired_cols = int(1.0 * c * (desired_rows / r))
    frame = cv2.resize(frame, dsize=(desired_cols, desired_rows))

    # Draw all people's skeleton
    skeleton_detector.draw(frame, man)

    # Draw bounding box and label of each person
    if len(dict_id2skeleton):
        for id, label in dict_id2label.items():
            skeleton = dict_id2skeleton[id]
            # scale the y data back to original
            skeleton[1::2] = skeleton[1::2] / scale_h

    # Add blank to the left for displaying prediction scores of each class
    frame = plots.add_space_to_information(frame)
    cv2.putText(frame, "Кадр:" + str(index_frame), (20, 20), fontScale=1.5, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 0, 0), thickness=2)

    # Draw predicting score for only 1 person
    if len(dict_id2skeleton):
        classifier_of_a_person = person_classifier.get_man_classifier(id='min')
        classifier_of_a_person.draw_score_on_frame(frame)
    return frame

def readVideo(images_loader, skeleton_detector, multiperson_tracker, multiperson_classifier, img_displayer, video_writer, DST_FOLDER, DST_SKELETON_FOLDER_NAME, SKELETON_FILENAME_FORMAT, img_disp_desired_rows, network_manager):
    try:
        network_manager.start_detection()
        ith_img = -1
        while images_loader.has_frame():
            img = images_loader.read_frame()
            # img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            ith_img += 1
            frame = img.copy()
            print(f"\nОбработка {ith_img} изображения ...")

            # -- Detect skeletons
            humans = skeleton_detector.detect(img)
            skeletons, scale_h = skeleton_detector.people_to_skeletons(humans)
            skeletons = remove_bad_skeletons(skeletons)

            # -- Track people
            dict_id2skeleton = multiperson_tracker.track(skeletons)  # int id -> np.array() skeleton

            # -- Recognize action of each person
            if len(dict_id2skeleton):
                dict_id2label = multiperson_classifier.classify(dict_id2skeleton)

            # -- Draw
            frame = draw_result_frame(
                frame, ith_img, humans, dict_id2skeleton, skeleton_detector,
                multiperson_classifier, img_disp_desired_rows, dict_id2label, scale_h
            )

            # Print label of a person
            if len(dict_id2skeleton):
                min_id = min(dict_id2skeleton.keys())
                action = dict_id2label[min_id]
                print(f"Распознано действие: {action}")
                network_manager.detected_action(action)

            # -- Display image, and write to video.avi
            img_displayer.show_frame(frame)
            video_writer.write(frame)

            # -- Get skeleton data and save to file
            skels_to_save = save_skeleton_data(dict_id2label, dict_id2skeleton)
            save_list(DST_FOLDER + DST_SKELETON_FOLDER_NAME +
                      SKELETON_FILENAME_FORMAT.format(ith_img), skels_to_save)

        # Ensure finalization is done after the loop
        video_writer.stop()
        network_manager.stop_detection()
        print("Program ends")
    finally:
        video_writer.stop()
        network_manager.stop_detection()
        print("Program ends")


def init_settings_and_run():
    CLASSES = np.array(Config.CLASSES)
    SKELETON_FILENAME_FORMAT = Config.SKELETON_FORMAT
    DST_VIDEO_FPS = Config.OUTPUT_VIDEO_FPS
    DST_VIDEO_NAME = Config.OUTPUT_VIDEO_NAME
    DST_FOLDER = f"output/{Config.DST_FOLDER_NAME}/"
    DST_SKELETON_FOLDER_NAME = Config.SKELETON_FOLDER_NAME

    # Настройки для OPENPOSE
    OPENPOSE_MODEL = Config.OPENPOSE_MODEL
    OPENPOSE_IMG_SIZE = Config.IMAGE_SIZE
    WINDOW_SIZE = Config.WINDOWS_SIZE

    # Настройки для вывода
    DESIRED_ROWS = Config.DESIRED_ROWS

    # Инициализация детектора, трекера и классификатора
    person_tracker = Tracker()
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    person_classifier = PersonClassifier()
    network_manager = NetworkManager()

    for videoname in ['hello.mp4', 'swipe_right.mp4', 'swipe_left.mp4', 'ready.mp4', 'cross.mp4']:
        # Загрузчик входных данных и интерфейс
        images_loader = get_video_reader(Config.SRC_DATA_TYPE, videoname)
        # images_loader = get_video_reader(Config.SRC_DATA_TYPE, Config.SRC_DATA_PATH)
        img_displayer = lib_images_io.ImageDisplayer()

        # Папки для выходных данных
        os.makedirs(DST_FOLDER, exist_ok=True)
        os.makedirs(DST_FOLDER + DST_SKELETON_FOLDER_NAME, exist_ok=True)

        # Запись видео работы программы
        video_writer = lib_images_io.VideoWriter(DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FPS)

        readVideo(images_loader, skeleton_detector, person_tracker, person_classifier, img_displayer,
                  video_writer, DST_FOLDER, DST_SKELETON_FOLDER_NAME, SKELETON_FILENAME_FORMAT,
                  DESIRED_ROWS, network_manager)

        time.sleep(5)

if __name__ == '__main__':
    # clear cuda memory костыль жесть, но а что поделать
    from numba import cuda
    device = cuda.get_current_device()
    device.reset()

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ''))  # fix dir

    init_settings_and_run()

