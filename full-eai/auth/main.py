import sqlite3
import tkinter as tk
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import pygame
import os 
import subprocess
import time

api = "Pk3HdsDu0xhp7OKVG_5xl02rvHOc4vWEDuFRPjZB6m3p"
url = "https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/51e11911-183c-4c68-8bc0-a233c527f414"

authenticator = IAMAuthenticator(api)
tts = TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(url)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def db_create():
    try:
        conn = sqlite3.connect('admins.db')
        conn.execute('''CREATE TABLE admin
                (ID INTEGER PRIMARY KEY AUTOINCREMENT,
                fullname   TEXT    NOT NULL UNIQUE,
                password    TEXT     NOT NULL);''')
    except Exception as e:
        print(e)

def db_insert():
    try:
        conn = sqlite3.connect('admins.db')
        conn.execute("INSERT INTO admin (fullname,password) \
        VALUES ('admin', 'admin');")
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)

#make sure to uncomment these if you run the db first time
db_create()
db_insert()

window = tk.Tk('Authentication')
window.title("Authentication")
window.geometry("400x200")

nameLabel = tk.Label(text="Full Name")
nameEntry = tk.Entry(width=50, bg="#ededed")
nameEntry.insert(0, "Enter your name")
nameEntry.configure(state=tk.DISABLED)

def nameonClick(event):
    nameEntry.configure(state=tk.NORMAL)
    nameEntry.delete(0, tk.END)
    nameEntry.unbind('<Button-1>', nameonClickId)

nameonClickId = nameEntry.bind('<Button-1>', nameonClick)

passwordLabel = tk.Label(text="Password")
passwordEntry = tk.Entry(width=50, bg="#ededed", show="*")
passwordEntry.insert(0, "Password")
passwordEntry.configure(state=tk.DISABLED)

def submit():
    fullname = nameEntry.get()
    password = passwordEntry.get()
    try:
        conn = sqlite3.connect('admins.db')
        cursor = conn.cursor()
        cursor.execute(f"""SELECT * FROM admin WHERE fullname="{fullname}" AND password="{password}";""")
        results = cursor.fetchall()

        if len(results) < 1:
            print('User does not exist in the database.')
            # with open('sounds/user-not-exist.mp3', 'wb') as audio_file:
            #     res = tts.synthesize("Credentials are not correct, check them and try again.", accept='audio/mp3',
            #                          voice='en-US_AllisonV3Voice').get_result()
            #     audio_file.write(res.content)
            pygame.mixer.init()
            pygame.mixer.music.load(os.path.join(BASE_DIR, 'auth/sounds/user-not-exist.mp3'))
            pygame.mixer.music.play()
        else:
            # then we can go to new window
            window.destroy()
            print(results[0][2])
            with open('sounds/user-hey.mp3', 'wb') as audio_file:
                res = tts.synthesize(f"Hello there, {results[0][2]}. You will be redirected to login and register page now.", accept='audio/mp3',
                                     voice='en-US_AllisonV3Voice').get_result()
                audio_file.write(res.content)
            pygame.mixer.init()
            pygame.mixer.music.load(os.path.join(BASE_DIR, 'sounds/user-hey.mp3'))
            pygame.mixer.music.play()
            for row in results:
                print(row)
            time.sleep(7)
            subprocess.call(['python', './main.py'])
            window.destroy()
            subprocess.call(['python', './main.py'])
    except Exception as e:
        print(e)
    
def passwordonClick(event):
    passwordEntry.configure(state=tk.NORMAL)
    passwordEntry.delete(0, tk.END)
    passwordEntry.unbind('<Button-1>', passwordonClickId)

passwordonClickId = passwordEntry.bind('<Button-1>', passwordonClick)

nameLabel.pack()
nameEntry.pack()
passwordLabel.pack()
passwordEntry.pack()

loginButton = tk.Button(
    text="Login",
    width=25,
    height=3,
    command=submit
)

loginButton.pack()
window.mainloop()


