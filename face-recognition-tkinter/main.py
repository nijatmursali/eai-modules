import tkinter as tk
import speech_recognition as sr
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import pygame
import time
import os
from re import search

window = tk.Tk()
window.title("EAI Modules")
window.geometry('400x200')
window.configure(background='gold')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
pygame.mixer.music.load('sounds/speech.mp3')
pygame.mixer.music.play()
pygame.mixer.music.queue('sounds/beep.mp3')
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
        loginWindow()

    elif (wantSignup == True):
        pygame.mixer.init()
        pygame.mixer.music.load('sounds/speech-signup.mp3')
        pygame.mixer.music.play()
        registerWindow()


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


a = tk.Button(text="LOGIN", width=5, height=2, command=loginWindow)
b = tk.Button(text="REGISTER", width=5, height=2, command=registerWindow)
c = tk.Button(text="STOP ASSISTANT", command=stop)
photo = tk.PhotoImage(file=r"./images/mic.png")
photo = photo.subsample(10, 10)
tk.Button(window, text='Listen', image=photo,
          command=listen).pack(side=tk.BOTTOM)


# we could also use a.pack(side=tk.LEFT) etc
a.place(relx=0.3, rely=0.5, anchor=tk.CENTER)
b.place(relx=0.7, rely=0.5, anchor=tk.CENTER)
c.place(relx=0.15, rely=0.1, anchor=tk.CENTER)
window.mainloop()
