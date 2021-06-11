# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:57:32 2021

@author: Jonathan
"""
import json
import requests
import numpy as np
import time
import pandas as pd
import math
import psycopg2



#request data - one day, 30 min bars, ohlc - 13 rows
def getPolygonData(after, ticker):
    APIkey = polyKEY
    min30 = 1800000
    oneDay = 86400000
    timeFrame = (oneDay*5)
    def fire_away(after=after):
        before = int(after)+timeFrame
        before = str(before)
        url = 'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/30/minute/{after}/{before}?unadjusted=false&apiKey={APIkey}'.format(ticker=ticker, after=after, before=before, APIkey=APIkey)
        print(url)
        response = requests.get(url)
        assert response.status_code == 200
        data = json.loads(response.text)
        return data
    current_tries = 1
    while current_tries < 6:
        try:
            time.sleep(.2)
            response = fire_away()
            return response
        except:
            time.sleep(.2)
            current_tries += 1
    return fire_away()

#loop through and record 13 lines of data for each day in given time period
def recordData(after, last, ticker):
    start_time = time.time()
    count = 0
    opn, high, low, close, volume, epochDate = [], [], [], [], [], []
    df = pd.DataFrame()
    min30 = 1800000
    oneDay = 86400000
    while int(after) < int(last):
        tries = 0
        while tries < 2:
            data = getPolygonData(after=after, ticker=ticker)
            #ensure it is a full day of trading (no weekends, half days, or holidays)
            if data['resultsCount'] < 1:    
                tries += 1
                time.sleep(10)
            else:
                z = 0
                length = data['resultsCount']
                #record data
                while z < length:
                    opn.append(data['results'][z]['o'])
                    high.append(data['results'][z]['h'])
                    low.append(data['results'][z]['l'])
                    close.append(data['results'][z]['c'])
                    volume.append(data['results'][z]['v'])
                    epochDate.append(data['results'][z]['t'])
                    z+=1
                break
        count
        count+=1
        print(count)    
        after = int(after)
        after+=(oneDay*5)
        after = str(after)
        time.sleep(12) 
    #copy to dataframe    
    df['opn'] = opn
    df['high'] = high
    df['low'] = low
    df['close'] = close
    df['volume'] = volume
    df['epochDate'] = epochDate
    df['date'] = pd.to_datetime(df['epochDate'], unit='ms').dt.tz_localize('utc').dt.tz_convert('America/Los_Angeles')    
    end_time = time.time()
    totalTime = (end_time-start_time)/60

    return df, totalTime


def get_db_connection():
    conn = psycopg2.connect(
    host="host",
    database="database",
    user="user",
    password="password")
    cur = conn.cursor()
    return conn, cur

def insertNew(df):
    conn, cur = get_db_connection()
    data = df.to_numpy()
    cur.executemany("INSERT INTO stocks VALUES(%s,%s,%s,%s,%s,%s,%s,%s)", data)
    conn.commit()
    cur.close()
    conn.close()

def updateDB():
    tickers = ['AAPL','AMZN','GOOG','IBM','GS']
    for i in tickers:
        #epoch = getLastEpoch(i)[0][0]
        #cur30Time = math.floor((time.time()*1000)/1800000)*1800000
        #df, totalTime = recordData(str(epoch), str(cur30Time), i) 
        df, totalTime = recordData('1619962200000', '1622554200000', i)
        df['stock'] = i
        df.drop_duplicates(subset ="epochDate", keep = 'last', inplace = True)
        df = df[1:]
        insertNew(df)
        
def createDB():
    conn, cur = get_db_connection()
    cur.execute('DROP table if exists stocks')
    cur.execute('create table if not exists stocks(opn real, high real, low real, close real, volume bigint, epochDate bigint, date text, stock text)')
    conn.commit()
    cur.close()
    conn.close()
    
    
createDB()
    
updateDB()
