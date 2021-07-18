import tkinter as tk
import speech_recognition as sr
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import pygame
import time
import os
from subprocess import call
from re import search

# some declarations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

# functions to call the libraries

def call_mario():
    call(['python', os.path.join(BASE_DIR, 'full-eai/ra/mario/Test.py')])

def call_mountain():
    call(['python', os.path.join(BASE_DIR, 'full-eai/ra/mountain-car/main.py')])

def call_login():
    call(['python', os.path.join(BASE_DIR, 'full-eai/cnn/login.py')])
def call_register():
    call(['python', os.path.join(BASE_DIR, 'full-eai/cnn/register.py')])

# preparing the database (sqlite) for adding the names and etc

window = tk.Tk()
window.title("EAI Modules")
window.geometry('400x200')
window.configure(background='gold')

api = "Pk3HdsDu0xhp7OKVG_5xl02rvHOc4vWEDuFRPjZB6m3p"
url = "https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/51e11911-183c-4c68-8bc0-a233c527f414"

authenticator = IAMAuthenticator(api)
tts = TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(url)

# with open('sounds/speech-signup-welcome.mp3', 'wb') as audio_file:
#     res = tts.synthesize("Hello again. As you are new user let's get your video for vision processing to detect your face and register you to the system.", accept='audio/mp3',
#                          voice='en-US_AllisonV3Voice').get_result()
#     audio_file.write(res.content)
pygame.mixer.init()
pygame.mixer.music.load(os.path.join(BASE_DIR, 'full-eai/sounds/speech.mp3'))
pygame.mixer.music.play()
pygame.mixer.music.queue(os.path.join(BASE_DIR, 'full-eai/sounds/beep.mp3'))
mic = sr.Microphone()


def listen():
    r = sr.Recognizer()
    with mic as source:
        audio = r.listen(source)
    text = r.recognize_google(audio)

    if search("login", text) or search("sign in", text):
        wantLogin = True
        wantSignup = False
    elif search("register", text) or search("sign up", text) or search("signup", text):
        wantSignup = True
        wantLogin = False
    if(wantLogin == True):
        pygame.mixer.init()
        pygame.mixer.music.load('sounds/speech-login.mp3')
        pygame.mixer.music.play()
        call_login()
        #loginWindow()

    elif (wantSignup == True):
        pygame.mixer.init()
        pygame.mixer.music.load('sounds/speech-signup.mp3')
        pygame.mixer.music.play()
        call_register()
        #registerWindow()


def stop():
    pygame.mixer.music.stop()


def open_camera():
    pass


def loginWindow():
    loginWindow = tk.Toplevel(window)
    loginWindow.title("Login")
    loginWindow.geometry('400x200')
    loginWindow.configure(background='gold')
    time.sleep(5)
    pygame.mixer.init()
    pygame.mixer.music.load('sounds/speech-login-welcome.mp3')
    pygame.mixer.music.play()


def registerWindow():
    registerWindow = tk.Toplevel(window)
    registerWindow.title("Register")
    registerWindow.geometry('400x200')
    registerWindow.configure(background='gold')
    time.sleep(6)
    pygame.mixer.init()
    pygame.mixer.music.load('sounds/speech-signup-welcome.mp3')
    pygame.mixer.music.play()


mic_image = tk.PhotoImage(file=os.path.join(BASE_DIR, 'full-eai/images/mic.png'))
mic_image = mic_image.subsample(10, 10)


def listen_home():
    r = sr.Recognizer()
    with mic as source:
        audio = r.listen(source)
    text = r.recognize_google(audio)
    wantMario = False
    wantMountainCar = False

    if search("mountain", text) or search("car", text) or search("mountain car", text):
        wantMountainCar = True
        wantMario = False
    elif search("mario", text) or search("Mario", text):
        wantMario = True
        wantMountainCar = False
    if(wantMountainCar == True):
        # with open('sounds/speech-mountain-car.mp3', 'wb') as audio_file:
        #     res = tts.synthesize("Great choice. In this implementation you can see the graphs and the car trying to move upwards to the flag. We have received 90 percent accuracy for this project.", accept='audio/mp3',
        #                          voice='en-US_AllisonV3Voice').get_result()
        #     audio_file.write(res.content)
        pygame.mixer.init()
        pygame.mixer.music.load('sounds/speech-mountain-car.mp3')
        pygame.mixer.music.play()
        # mountainCarWindow()

    elif (wantMario == True):
        # with open('sounds/speech-mario.mp3', 'wb') as audio_file:
        #     res = tts.synthesize("I see you selected Mario. Good choice. In this implementation you can see the graphs and the mario trying to move to the finish where the flag is. We have received 90 percent accuracy for this project.", accept='audio/mp3',
        #                          voice='en-US_AllisonV3Voice').get_result()
        #     audio_file.write(res.content)
        pygame.mixer.init()
        pygame.mixer.music.load('sounds/speech-mario.mp3')
        pygame.mixer.music.play()
        # marioWindow()

    print(text)


def homeWindow():
    homeWindow = tk.Toplevel(window)
    homeWindow.title("Home")
    homeWindow.geometry('400x200')
    homeWindow.configure(background='gold')
    # with open('sounds/speech-home.mp3', 'wb') as audio_file:
    #     res = tts.synthesize("Good to see you here. We have 2 implentation for the reinforcement learning. One is Mountain Car and another one is Mario. Which one you want to choose?", accept='audio/mp3',
    #                          voice='en-US_AllisonV3Voice').get_result()
    #     audio_file.write(res.content)
    pygame.mixer.init()
    pygame.mixer.music.load('sounds/speech-home.mp3')
    pygame.mixer.music.play()
    a = tk.Button(homeWindow, text="MOUNTAIN CAR",
                  width=10, height=2)
    b = tk.Button(homeWindow, text="MARIO", width=5,
                  height=2)
    tk.Button(homeWindow, text='Listen', image=mic_image,
              command=listen_home).pack(side=tk.BOTTOM)
    # we could also use a.pack(side=tk.LEFT) etc
    a.place(relx=0.3, rely=0.5, anchor=tk.CENTER)
    b.place(relx=0.7, rely=0.5, anchor=tk.CENTER)


a = tk.Button(text="LOGIN", width=5, height=2, command=loginWindow)
b = tk.Button(text="REGISTER", width=5, height=2, command=registerWindow)
d = tk.Button(text="HOME", width=5, height=2, command=call_mario)
c = tk.Button(text="STOP ASSISTANT", command=stop)
photo = tk.PhotoImage(file=os.path.join(BASE_DIR, 'full-eai/images/mic.png')) 
photo = photo.subsample(10, 10)
tk.Button(window, text='Listen', image=photo,
          command=listen).pack(side=tk.BOTTOM)


# we could also use a.pack(side=tk.LEFT) etc
a.place(relx=0.2, rely=0.5, anchor=tk.CENTER)
b.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
d.place(relx=0.8, rely=0.5, anchor=tk.CENTER)
c.place(relx=0.15, rely=0.1, anchor=tk.CENTER)
window.mainloop()
