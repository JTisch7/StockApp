# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:44:23 2021

@author: Jonathan
"""

import flask
from flask import request, jsonify, render_template, url_for, flash, redirect, session
from werkzeug.exceptions import abort

import sklearn
import json
import pickle
import numpy as np
import pandas as pd
import time
import psycopg2
from dateutil.parser import parse
#import asyncio
import os
import requests
import math
import threading
from polygonPull import getPolygonData, recordData
import datetime

#from waitress import serve

app = flask.Flask(__name__)
#app.config["DEBUG"] = True
app.config['SECRET_KEY'] = '123453452435243545455678'
APIkey = os.environ['polyKEY']

def get_db_connection():
    DATABASE_URL = os.environ['DATABASE_URL']
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    return conn, cur

def getLastEpoch(stk):
    conn, cur = get_db_connection()
    cur.execute("SELECT max(epochDate) FROM stocks WHERE stock=%s",(stk,))
    epoch = cur.fetchall()
    cur.close()
    conn.close()
    return epoch

def insertNew(df):
    conn, cur = get_db_connection()
    data = df.to_numpy()
    cur.executemany("INSERT INTO stocks VALUES(%s,%s,%s,%s,%s,%s,%s,%s)", data)
    conn.commit()
    cur.close()
    conn.close()


@app.before_first_request
def startingFunc():
    def updateDB():
        tickers = ['AAPL','AMZN','GOOG','IBM','GS']
        for i in tickers:
            epoch = getLastEpoch(i)[0][0]
            cur30Time = math.floor((time.time()*1000)/1800000)*1800000
            df, totalTime = recordData(str(epoch), str(cur30Time), i) 
            df['stock'] = i
            df.drop_duplicates(subset ="epochDate", keep = 'last', inplace = True)
            df = df[1:]
            insertNew(df)
        
    thread = threading.Thread(target=updateDB)
    thread.start()


yPred = ''
yPredUp = ''
yPredDown = ''
yPred2 = ''
yPred2Up = ''
yPred2Down = ''
mod = ''
mod2 = ''
stocks = ''
stocksPred = ''
stockPred = ''
opn = ''
high = ''
low = ''
close = ''
stk = ''
frm = ''
to = ''

gbrtMod = pickle.load(open('models/gbrt.pkl','rb'))
adaMod = pickle.load(open('models/ada.pkl','rb'))
RFMod = pickle.load(open('models/RF.pkl','rb'))
SVCMod = pickle.load(open('models/SVC.pkl','rb'))
logRegMod = pickle.load(open('models/logReg.pkl','rb'))

def dataPipeline(opn, high, low, close, volPrior, volCurrent):
    vix=.002
    xTest = pd.DataFrame(columns = ['close_open','high_open','low_open','volPercent','vixClose'])
    x1 = np.log(close/opn)/vix
    x2 = np.log(high/opn)/vix
    x3 = np.log(opn/low)/vix
    x4 = np.log(volCurrent/volPrior)
    x5 = vix
    xTest.loc[0] = [x1,x2,x3,x4,x5]
    return xTest

def predict(xTest, model=gbrtMod):
    newPipeline = pickle.load(open('models/full_pipeline.pkl','rb'))
    transformedData = newPipeline.transform(xTest)      
    result = model.predict_proba(transformedData)
    return result

def setupStockCharts(stk, frm, to):
    conn, cur = get_db_connection()
    cur.execute("SELECT * FROM stocks WHERE stock=%s AND date::timestamp >= %s AND date::timestamp <= %s", (stk, frm, to))
    columns = ('opn', 'high', 'low', 'close', 'volume', 'epochDate', 'date', 'stock')
    results = []
    for row in cur.fetchall():
        results.append(dict(zip(columns, row)))
    cur.close()
    conn.close()        
    return results


curTime = datetime.datetime.now()
TenDays = datetime.timedelta(10)
OneMonth = datetime.timedelta(30)
stocks = setupStockCharts('AAPL', curTime-OneMonth, curTime)
stocksPred = setupStockCharts('AMZN', curTime-TenDays, curTime)


@app.route('/', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        if request.form.get('chart') == 'val1':
            global stk, frm, to
            stk = request.form['stock']
            frm = request.form['from']
            to = request.form['to']
        
            if (stk and frm and to):
                global stocks
                stocks = setupStockCharts(stk, frm, to)
            return redirect('#charts')

                
        if request.form.get('pred') == 'val2':
            global mod, stockPred
            mod = request.form['model']
            stockPred = request.form['stockPred']
            if (mod and stockPred):
                global gbrtMod, RFMod, logRegMod, adaMod, SVCMod
                if mod == 'GRADIENT BOOSTING':
                    mod = gbrtMod
                if mod == 'RANDOM FOREST':
                    mod = RFMod
                if mod == 'LOGISTIC REGRESSION CLASSIFIER':
                    mod = logRegMod
                if mod == 'ADABOOST':
                    mod = adaMod
                if mod == 'SUPPORT VECTOR MACHINES':
                    mod = SVCMod                
                global stocksPred
                stocksPred = setupStockCharts(f'{stockPred}',curTime-TenDays, curTime)
                global yPred, yPredUp, yPredDown
                xTest = dataPipeline(stocksPred[-1]['opn'],stocksPred[-1]['high'],stocksPred[-1]['low'],
                                     stocksPred[-1]['close'],stocksPred[-2]['volume'],stocksPred[-1]['volume'])
                yPred = predict(xTest, mod)
                yPredUp = round(yPred[0][1], 4)
                yPredDown = round(yPred[0][0], 4)
            return redirect('#chartPred')
        
        if request.form.get('predict') == 'val3':
            if (request.form['open'] and request.form['high'] and request.form['low'] and request.form['close'] and request.form['model2']):
                global mod2, opn, high, low, close
                opn = float(request.form['open'])
                high = float(request.form['high'])
                low = float(request.form['low'])   
                close = float(request.form['close'])            
                mod2 = request.form['model2']
            
                if mod2 == 'GRADIENT BOOSTING':
                    mod2 = gbrtMod
                if mod2 == 'RANDOM FOREST':
                    mod2 = RFMod
                if mod2 == 'LOGISTIC REGRESSION CLASSIFIER':
                    mod2 = logRegMod
                if mod2 == 'ADABOOST':
                    mod2 = adaMod
                if mod2 == 'SUPPORT VECTOR MACHINES':
                    mod2 = SVCMod   
                global yPred2, yPred2Up, yPred2Down
                xTest = dataPipeline(opn, high, low, close, 500, 400)
                yPred2 = predict(xTest, mod2)
                yPred2Up = round(yPred2[0][1], 4)
                yPred2Down = round(yPred2[0][0], 4)                
            return redirect('#predict')   
        
        
    return render_template('indexW.html', stocks=stocks, stocksPred=stocksPred, yPredUp=yPredUp, yPredDown=yPredDown, yPred2Up=yPred2Up, yPred2Down=yPred2Down, mod2=mod2, 
                           mod=mod, stockPred=stockPred, opn=opn, high=high, low=low, close=close, stk=stk, frm=frm, to=to)   


@app.route('/api/v1/stockdata', methods=['GET'])
def api_filter():
    query_parameters = request.args

    stock = query_parameters.get('stock')
    volume = query_parameters.get('volume')
    close = query_parameters.get('close')
    before = query_parameters.get('before')
    after = query_parameters.get('after')

    query = "SELECT * FROM stocks WHERE"
    to_filter = []

    if stock:
        query += ' stock=%s AND'
        to_filter.append(stock)
    if volume:
        query += ' volume>=%s AND'
        to_filter.append(volume)
    if close:
        query += ' close>=%s AND'
        to_filter.append(close)
    if before:
        query += ' epochDate<=%s AND'
        to_filter.append(before)
    if after:
        query += ' epochDate>=%s AND'
        to_filter.append(after)
    if not (stock or before or after):
        return page_not_found(404)

    query = query[:-4] + ';'

    conn, cur = get_db_connection()
    cur.execute(query, to_filter)
    columns = ('opn', 'high', 'low', 'close', 'volume', 'epochDate', 'date', 'stock')
    results = []
    for row in cur.fetchall():
        results.append(dict(zip(columns, row)))
    cur.close()
    conn.close()        
    return jsonify(results)


@app.route('/api/v1/predict', methods=['GET'])
def predictFunc():
    query_parameters = request.args

    mod = query_parameters.get('model')
    opn = float(query_parameters.get('open'))
    high = float(query_parameters.get('high'))
    low = float(query_parameters.get('low'))
    close = float(query_parameters.get('close'))
    if mod == 'gbrtMod':
        mod = gbrtMod
    if mod == 'RFMod':
        mod = RFMod
    if mod == 'logRegMod':
        mod = logRegMod
    if mod == 'adaMod':
        mod = adaMod
    if mod == 'SVCMod':
        mod = SVCMod  

    if not (mod and open and high and low and close):
        return page_not_found(404)

    xTest = dataPipeline(opn, high, low, close, 500, 400)
    return jsonify(list(predict(xTest, mod)[0]))


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

if __name__ == '__main__':
    app.run(threaded=True)
