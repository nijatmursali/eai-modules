import mtcnn
from tensorflow.keras.models import load_model
import dlib

import os
import pickle
from utils import *
from vision import Vision
from database import Credentials
from check import CheckGlasses
database_path = 'Database'
encodings_dict_path = os.path.join(database_path, 'encodings.pkl')
database_images = os.path.join(database_path, 'people')
database_names_path = os.path.join(database_path, 'usernames.txt')
shape_predictor_path = "data/shape_predictor_5_face_landmarks.dat"
encoder_model = './facenet_keras/data/model/facenet_keras.h5'
# encodings_path = './facenet_keras/data/encodings/encodings.pkl'



cap = cv2.VideoCapture(0)
shape_predictor = dlib.shape_predictor(shape_predictor_path)
frontal_face_detector = dlib.get_frontal_face_detector()
credentials = Credentials(database_names_path)


face_detector = mtcnn.MTCNN()
face_encoder = load_model(encoder_model)
# encoding_dict = load_pickle(encodings_dict_path)

checker = CheckGlasses(shape_predictor_path)
vision = Vision(credentials, database_path, database_images, cap, checker, frontal_face_detector, shape_predictor, face_encoder, face_detector, encodings_dict_path)

# print(vision.encode_dict)
# vision.handle_cv(True)
#
# c = cv2.VideoCapture(0)
# user_id = credentials.get_Username()
# vision.login(user_id)
user_id = credentials.write_database()
vision.register(user_id)
print(vision.encode_dict)