from scipy.spatial.distance import cosine

from utils import *
import os



class Vision:
    def __init__(self, database_path, cap, checker, detector, predictor, encoder, detector_login, encode_dict):
        self.database_path = database_path  # database of images

        self.detector = detector
        self.predictor = predictor
        self.cap = cap
        self.checker = checker
        self.encoder = encoder
        self.detector_login = detector_login
        self.encode_dict = encode_dict

    def save_image(self, username, img):
        blank = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
        rectan = cv2.rectangle(blank, (0, 200), (640, 270), (0, 255, 0), -1)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = 'Thank you for your registration!'
        textsize = cv2.getTextSize(text, font, 1, 2)
        textX = int((640 - (textsize[0][0])) / 2)
        textY = int((480 - (textsize[0][1])) / 2)
        cv2.putText(blank, 'Thank you for your registration!', (textX, textY), font, 1, (255, 255, 255), 2)
        cv2.imshow('image', blank)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

        file_name = username + '.jpg'
        cv2.imwrite(os.path.join(self.database_path, file_name), img)

    def handle_cv(self, image, username, x, y, glasses):
        taken = False
        red = (0, 0, 255)
        green = (0, 255, 0)
        font = cv2.FONT_HERSHEY_DUPLEX

        glasson = 'Glasses are detected!'
        glassoff = 'Glasses Free'
        glasson_warning = f'Please remove glasses!'
        glassoff_warning = f'Press Q when you are ready!'
        color = int(not glasses) * green + int(glasses) * red
        texton = int(not glasses) * glassoff + int(glasses) * glasson
        text = int(not glasses) * glassoff_warning + int(glasses) * glasson_warning
        cv2.putText(image, texton, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2, cv2.LINE_AA)
        rectan = cv2.rectangle(image, (0, 400), (640, 470), color, -1)

        textsize = cv2.getTextSize(text, font, 1, 2)
        textX = int((640 - (textsize[0][0])) / 2)
        cv2.putText(image, text, (textX, 445), font, 1, (255, 255, 255), 2)

    def capture_image(self, glasses, username):
        k = cv2.waitKey(10)

        if not glasses and k == ord('q'):
            if k == ord('q'):
                print('q basildi')
                _, img = cap.read()
                img = cv2.flip(img, 1)
                cv2.imshow('Result', img)
                cv2.waitKey(3000)

                cap.release()
                cv2.destroyAllWindows()
                self.save_image(username, img)

    def register(self, username):

        while self.cap.isOpened():
            # Read video frame
            _, img = self.cap.read()
            image = cv2.flip(img, 1)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # face detection
            rects = self.detector(gray, 1)
            # Face Detection
            for i, rect in enumerate(rects):
                # Get coordinates

                x_face = rect.left()
                y_face = rect.top()
                w_face = rect.right() - x_face
                h_face = rect.bottom() - y_face

                # Draw a border and add text annotations
                cv2.rectangle(img, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 0, 125), 2)
                # cv2.putText(img, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                #             (0, 255, 0), 2, cv2.LINE_AA)

                # Detect and mark landmarks
                landmarks = self.predictor(gray, rect)

                landmarks = self.checker.landmarks_to_np(landmarks)
                # for (x, y) in landmarks:
                #     cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

                # Linear regression
                LEFT_EYE_CENTER, RIGHT_EYE_CENTER = self.checker.get_centers(img, landmarks)

                # Face alignment
                aligned_face = self.checker.get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)

                self.handle_cv(img, username, x_face, y_face, self.checker.is_glasses_on(aligned_face))

                self.capture_image(self.checker.is_glasses_on(aligned_face), username)

                cv2.imshow("aligned_face #{}".format(i + 1), aligned_face)

            # show result
            cv2.imshow("Result", img)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:  # Press "Esc" to exit
                break

    def check_name_gen_text(self, username, image, condition, condition2, pt1, pt2, distance):
        green = (0, 255, 0)
        red = (0, 0, 255)
        white = (255, 255, 255)
        # color = tuple()
        # text = ''
        # name_signature = ''
        # coordinates = tuple

        if condition:
            color = red
            text = 'You are not that guy!'

            coordinates = (pt1[0], pt1[1] - 5)
            if condition2:  # name == unknown
                name_signature = 'unknown'
            else:  # name == database_name but not username
                name_signature = username
        else:
            color = green
            text = 'Welcome ' + username
            name_signature = username + f'{distance: .2f}'
            coordinates = pt1

        font = cv2.FONT_HERSHEY_DUPLEX

        textsize = cv2.getTextSize(text, font, 1, 2)
        textX = int((640 - (textsize[0][0])) / 2)

        cv2.rectangle(image, pt1, pt2, color, 2)
        cv2.putText(image, name_signature, coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        rectan = cv2.rectangle(image, (0, 400), (640, 470), color, -1)
        cv2.putText(image, text, (textX, 445), font, 1, white, 2)

    def login(self,
              username,
              recognition_t=0.5,
              confidence_t=0.99,
              required_size=(160, 160)):

        while self.cap.isOpened():
            ret, img = self.cap.read()
            if not ret:
                print('no frame')
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.detector_login.detect_faces(img_rgb)
            database_name = ''
            for res in results:
                if res['confidence'] < confidence_t:
                    continue
                face, pt_1, pt_2 = get_face(img_rgb, res['box'])
                encode = get_encode(self.encoder, face, required_size)
                encode = l2_normalizer.transform(encode.reshape(1, -1))[0]

                name = 'unknown'
                distance = float("inf")
                for db_name, db_encode in encoding_dict.items():
                    dist = cosine(db_encode, encode)
                    if dist < recognition_t and dist < distance:
                        name = db_name
                        distance = dist
                        database_name = db_name

                condition = (name.lower() != username.lower())
                condition2 = (name == 'unknown')
                print(name)
                print(username)
                print(database_name )
                print(condition)
                self.check_name_gen_text(name, img, condition, condition2, pt_1, pt_2, distance)
            cv2.imshow('camera', img)
            # print(name)

            if cv2.waitKey(1) & 0xFF == 27:
                break

