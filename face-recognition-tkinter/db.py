# here we will add the database stuff
import sqlite3
try:
    conn = sqlite3.connect('eai.db')
    print('Successfully connected to database')
except:
    print('Error has happened during execution!')
