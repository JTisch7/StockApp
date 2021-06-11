# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:35:23 2021

@author: Jonathan
"""

import numpy as np
import pandas as pd
import psycopg2



#create table from pd.dataframe
AAPLstock = pd.read_csv('C:/Users/Jonathan/commits/data/30MinAAPL2yrs.csv', index_col=(0))
AMZNstock = pd.read_csv('C:/Users/Jonathan/commits/data/30MinAMZN2yrs.csv', index_col=(0))
GOOGstock = pd.read_csv('C:/Users/Jonathan/commits/data/30MinGOOG2yrs.csv', index_col=(0))
IBMstock = pd.read_csv('C:/Users/Jonathan/commits/data/30MinIBM2yrs.csv', index_col=(0))
GSstock = pd.read_csv('C:/Users/Jonathan/commits/data/30MinGS2yrs.csv', index_col=(0))

AAPLstock['stock']='AAPL'
AMZNstock['stock']='AMZN'
GOOGstock['stock']='GOOG'
IBMstock['stock']='IBM'
GSstock['stock']='GS'

stocks = pd.concat([AAPLstock,AMZNstock,GOOGstock,IBMstock,GSstock]).reset_index(drop=True)

#create database
def moveData():
    conn = psycopg2.connect(
        host="host",
        database="database",
        user="user",
        password="password")
    
    cur = conn.cursor()
    cur.execute('DROP table if exists stocks')
    cur.execute('create table if not exists stocks(opn real, high real, low real, close real, volume bigint, epochDate bigint, date text, stock text, PRIMARY KEY (stock, epochDate))')
    data = stocks.to_numpy()
    cur.executemany("INSERT INTO stocks VALUES(%s, %s, %s, %s, %s, %s, %s, %s)", data)
    conn.commit()
    cur.close()
    conn.close()
    
moveData()









AAPLstock = pd.read_csv('C:/Users/Jonathan/commits/data/30MinAAPL2yrs.csv', index_col=(0))

def get_db_connection():
    conn = psycopg2.connect(
    host="host",
    database="database",
    user="user",
    password="password")
    cur = conn.cursor()
    return conn, cur

def getEpoch(stk):
    conn, cur = get_db_connection()
    cur.execute("SELECT max(epochDate) FROM stocks WHERE stock=%s",(stk,))
    epoch = cur.fetchall()
    cur.close()
    conn.close()
    return epoch

epoch = getEpoch('AAPL')[0][0]


