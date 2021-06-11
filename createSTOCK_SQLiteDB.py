# -*- coding: utf-8 -*-
"""
Created on Fri May 21 11:02:37 2021

@author: Jonathan
"""

#establish connection and create database
import sqlite3
from sqlite3 import Error

def sql_connection():
    try:
        con = sqlite3.connect('stockDB.db')
        return con
    except Error:
        print(Error)
        
con = sql_connection()

#create table from pd.dataframe
import pandas as pd
import numpy as np

AAPLstock = pd.read_csv('C:/Users/Jonathan/commits/data/30MinAAPL2yrs.csv', index_col=(0))
AMZNstock = pd.read_csv('C:/Users/Jonathan/commits/data/30MinAMZN2yrs.csv', index_col=(0))
GOOGstock = pd.read_csv('C:/Users/Jonathan/commits/data/30MinGOOG2yrs.csv', index_col=(0))

AAPLstock['stock']='AAPL'
AMZNstock['stock']='AMZN'
GOOGstock['stock']='GOOG'

stocks = pd.concat([AAPLstock,AMZNstock,GOOGstock])

def sql_fetch(con, stocks):
    cursorObj = con.cursor()
    cursorObj.execute('create table if not exists stocks(opn real, high real, low real, close real, volume integer, epochDate integer, date integer, stock text)')
    data = np.array(stocks[:])
    cursorObj.executemany("INSERT INTO stocks VALUES(?, ?, ?, ?, ?, ?, ?, ?)", data)
    con.commit()
    
sql_fetch(con, stocks)

'''
#delete table
def sql_fetch(con):
    cursorObj = con.cursor()
    cursorObj.execute('DROP table if exists AAPLstock')
    cursorObj.execute('DROP table if exists AMZNstock')
    cursorObj.execute('DROP table if exists GOOGstock')
    cursorObj.execute('DROP table if exists stocks')
    con.commit()
sql_fetch(con)
'''