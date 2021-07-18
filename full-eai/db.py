# here we will add the database stuff
import sqlite3
import os

DATABASE_NAME = "eai.db"
try:
    folders = os.listdir()
    if(DATABASE_NAME not in folders):
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()

        sql = """CREATE TABLE users(
		          id integer unique primary key autoincrement,
		          name text)
		      """
        c.executescript(sql)
        conn.commit()
        conn.close()
    print('Successfully connected to database')
except:
    print('Error has happened during execution!')
