# -*- coding: utf-8 -*-
"""
Created on Thu May 20 14:48:27 2021

@author: Jonathan
"""

#establish connection and create database
import sqlite3
from sqlite3 import Error

def sql_connection():
    try:
        con = sqlite3.connect('mydatabase.db')
        return con
    except Error:
        print(Error)

def sql_table(con):
    cursorObj = con.cursor()
    cursorObj.execute("CREATE TABLE employees(id integer PRIMARY KEY, name text, salary real, department text, position text, hireDate text)")
    con.commit()

con = sql_connection()
sql_table(con)

#random queries
cursorObj = con.cursor()
cursorObj.execute("INSERT INTO employees VALUES(1, 'John', 700, 'HR', 'Manager', '2017-01-04')")
con.commit()

entities = (2, 'Andrew', 800, 'IT', 'Tech', '2018-02-06')
cursorObj.execute('''INSERT INTO employees(id, name, salary, department, position, hireDate) VALUES(?, ?, ?, ?, ?, ?)''', entities)

cursorObj.execute('UPDATE employees SET name = "Rogers" where id = 2')
cursorObj.execute('SELECT * FROM employees ')
cursorObj.execute('SELECT id, name FROM employees')

#print results
def sql_fetch(con):
    cursorObj = con.cursor()
    cursorObj.execute('SELECT * FROM employees')
    rows = cursorObj.fetchall()
    for i, row in enumerate(rows):
        if i == 1:
            print(i, row)
    #or
    #[print(row) for row in cursorObj.fetchall()]

sql_fetch(con)

#print results from where statement
def sql_fetch(con):
    cursorObj = con.cursor()
    cursorObj.execute('SELECT id, name, salary FROM employees WHERE salary > 500.0')
    rows = cursorObj.fetchall()
    for row in rows:
        print(row)
sql_fetch(con)

#print num of rows
rows = cursorObj.fetchall()
print(len(rows))

#create table from pd.dataframe
import pandas as pd
import numpy as np
stock = pd.read_csv('C:/Users/Jonathan/commits/data/30MinAAPL2yrs.csv', index_col=(0))

con = sqlite3.connect('stockDB.db')

def sql_fetch(con):
    cursorObj = con.cursor()
    cursorObj.execute('create table if not exists AAPLstock(opn real, high real, low real, close real, volume integer, epochDate integer, date integer)')
    data = np.array(stock[:])
    cursorObj.executemany("INSERT INTO AAPLstock VALUES(?, ?, ?, ?, ?, ?, ?)", data)
    con.commit()
sql_fetch(con)

#create table from table
def sql_fetch(con):
    cursorObj = con.cursor()
    cursorObj.execute('create table AAPLhighVol as SELECT * FROM AAPLstock WHERE volume > 60000000')
    con.commit()
sql_fetch(con)

#query table and add results to list
x =[]
def sql_fetch(con):
    cursorObj = con.cursor()
    cursorObj.execute('SELECT * FROM AAPLstock WHERE volume > 60000000')
    [x.append(row) for row in cursorObj.fetchall()]
    [print(row) for row in cursorObj.fetchall()]
sql_fetch(con)


#delete table
con = sqlite3.connect('stockDB.db')

def sql_fetch(con):
    cursorObj = con.cursor()
    cursorObj.execute('DROP table if exists AAPLstock')
    con.commit()
sql_fetch(con)

#list table names
def sql_fetch(con):
    cursorObj = con.cursor()
    cursorObj.execute('SELECT name from sqlite_master where type= "table"')
    print(cursorObj.fetchall())
sql_fetch(con)