
import psycopg2
import pandas as pd
import numpy as np
import sqlalchemy


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
    conn, cur = get_db_connection()
    cur.execute('DROP table if exists stocks')
    cur.execute('create table if not exists stocks(opn real, high real, low real, close real, volume bigint, epochDate bigint, date text, stock text)')
    data = stocks.to_numpy()
    cur.executemany("INSERT INTO stocks VALUES(%s, %s, %s, %s, %s, %s, %s, %s)", data)
    conn.commit()
    cur.close()
    conn.close()
    
moveData()
    




'''

\/ \/ \/  OTHER FUNTIONS TO HELP  \/ \/ \/

'''


#delete a table
def deleteTable():
    conn, cur = get_db_connection()
    cur.execute('DROP table if exists test')
    conn.commit()
    cur.close()
    conn.close()
    
deleteTable()

#dealing with datetime/timestamp in postgreSQL/python
from dateutil.parser import parse
z = '02-26-2020 10:00'
z= parse(z)

def test():
    conn, cur = get_db_connection()
    cur.execute('SELECT * FROM stocks WHERE date >= %s',(z,))
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data
    
test = test()


#connect to db
def get_db_connection():
    conn = psycopg2.connect(
    host="host",
    database="database",
    user="user",
    password="password")
    cur = conn.cursor()
    return conn, cur


#way to dictionarize or jsonify results for charts etc
import json

def setupStocks():
    conn, cur = get_db_connection()
    z = '2020-02-08 10:30'
    z1 = '2020-10-25 10:30'
    cur.execute("SELECT * FROM stocks WHERE stock='AAPL' AND date::timestamp >= %s AND date::timestamp <= %s", (z, z1))
    #stocks = cur.fetchall()
    
    columns = ('opn', 'high', 'low', 'close', 'volume', 'epochDate', 'date', 'stock')
    results = []
    for row in cur.fetchall():
        results.append(dict(zip(columns, row)))
        
    cur.close()
    conn.close()
    return json.dumps(results), results

stocks, stocks2= setupStocks()




