import copy
import csv
import itertools
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        img_width,
        img_height,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.img_width = img_width
        self.img_height = img_height

        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    def __call__(self, landmarks):
        landmark_list = self.calc_landmark_list(landmarks)
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = np.squeeze(self.interpreter.get_tensor(output_details_tensor_index))
        result_index = np.argmax(np.squeeze(result))
        probability = result[result_index]

        if probability < 0.2:
            result_index == -1

        return self.get_gesture_label(result_index)

    def get_gesture_label(self, hand_sign_id):
        if hand_sign_id == -1:
            return 'None'
        label = self.keypoint_classifier_labels[hand_sign_id]
        return label

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def calc_landmark_list(self, landmarks):
        landmark_point = []

        for _, landmark in enumerate(landmarks):
            landmark_x = min(int(landmark.x * self.img_width), self.img_width - 1)
            landmark_y = min(int(landmark.y * self.img_height), self.img_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return self.pre_process_landmark(landmark_point)
