import mtcnn
from keras.models import load_model
import dlib
from check import CheckGlasses
from database import Credentials
import cv2
import os
import pickle
from utils import *
from vision import Vision
from database import Credentials
from check import CheckGlasses
path = "data/shape_predictor_5_face_landmarks.dat"
database_path = 'Database'
database_names_path = os.path.join(database_path, 'usernames.txt')
cap = cv2.VideoCapture(0)
predictor = dlib.shape_predictor(path)
detector = dlib.get_frontal_face_detector()
credentials = Credentials(database_names_path)
encoder_model = './facenet_keras/data/model/facenet_keras.h5'
encodings_path = './facenet_keras/data/encodings/encodings.pkl'

face_detector = mtcnn.MTCNN()
face_encoder = load_model(encoder_model)
encoding_dict = load_pickle(encodings_path)

checker = CheckGlasses(path)
vision = Vision(database_path, cap, checker, detector, predictor, face_encoder, face_detector, encoding_dict)
# vision.handle_cv(True)
# user_id = credentials.write_database()
# c = cv2.VideoCapture(0)
user_id = credentials.get_Username()
vision.login(user_id)
# vision.register(user_id)
