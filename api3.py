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
import os
import requests
import math
import threading
from polygonPull import getPolygonData, recordData
import datetime

#from waitress import serve

app = flask.Flask(__name__)
secret_key = os.environ['secret']
app.config['SECRET_KEY'] = secret_key
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
'''
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
'''
def setupStockCharts(stk, frm, to):
    conn, cur = get_db_connection()
    cur.execute("SELECT * FROM stocks WHERE stock=%s AND epochdate >= %s AND epochdate <= %s", (stk, frm, to))
    columns = ('opn', 'high', 'low', 'close', 'volume', 'epochDate', 'date', 'stock')
    results = []
    for row in cur.fetchall():
        row = list(row)
        row.pop(6)
        row[5]=int(row[5])
        #row[6] = datetime.datetime.fromtimestamp(row[6])
        results.append(dict(zip(columns, row)))
    cur.close()
    conn.close()        
    return results


curTime = datetime.datetime.now()
TenDays = datetime.timedelta(10)
OneMonth = datetime.timedelta(30)
#stocks = setupStockCharts('AAPL', curTime-OneMonth, curTime)
#stocksPred = setupStockCharts('AMZN', curTime-TenDays, curTime)
stocks = setupStockCharts('AAPL', 1619499209000, 1630040009000)


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
                url = url_for('api_filter', _external=True, stock=stk, after=frm, before=to)
                response = requests.get(url)
                stocks = json.loads(response.text)
            return redirect('#charts')
    
        if request.form.get('pred') == 'val2':
            global mod, stockPred
            model = request.form['model']
            stockPred = request.form['stockPred']
            if (model and stockPred):
                if model == 'gbrtMod':
                    mod = gbrtMod
                if model == 'RFMod':
                    mod = RFMod
                if model == 'logRegMod':
                    mod = logRegMod
                if model == 'adaMod':
                    mod = adaMod
                if model == 'SVCMod':
                    mod = SVCMod 

                global stocksPred, yPred, yPredUp, yPredDown
                url = url_for('api_filter', _external=True, stock=stockPred, after=curTime-TenDays, before=curTime)
                response = requests.get(url)
                stocksPred = json.loads(response.text) 

                url2 = url_for('predictFunc', _external=True, model=model, open=stocksPred[-1]['opn'], high=stocksPred[-1]['high'], low=stocksPred[-1]['low'], close=stocksPred[-1]['close'])
                response2 = requests.get(url2)
                yPred = json.loads(response2.text)   
                yPredUp = round(yPred[0], 4)
                yPredDown = round(yPred[1], 4)

            return redirect('#chartPred')
        
        if request.form.get('predict') == 'val3':
            if (request.form['open'] and request.form['high'] and request.form['low'] and request.form['close'] and request.form['model2']):
                global mod2, opn, high, low, close
                opn = request.form['open']
                high = request.form['high']
                low = request.form['low']  
                close = request.form['close']            
                model = request.form['model2']
            
                if model == 'gbrtMod':
                    mod2 = gbrtMod
                if model == 'RFMod':
                    mod2 = RFMod
                if model == 'logRegMod':
                    mod2 = logRegMod
                if model == 'adaMod':
                    mod2 = adaMod
                if model == 'SVCMod':
                    mod2 = SVCMod 

                global yPred2, yPred2Up, yPred2Down
                url2 = url_for('predictFunc', _external=True, model=model, open=opn, high=high, low=low, close=close)
                response2 = requests.get(url2)
                yPred2 = json.loads(response2.text)   
                yPred2Up = round(yPred2[0], 4)
                yPred2Down = round(yPred2[1], 4)
     
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
        query += ' date::timestamp<=%s AND'
        to_filter.append(before)
    if after:
        query += ' date::timestamp>=%s AND'
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

    model = query_parameters.get('model')
    opn = float(query_parameters.get('open'))
    high = float(query_parameters.get('high'))
    low = float(query_parameters.get('low'))
    close = float(query_parameters.get('close'))
    if model == 'gbrtMod':
        model = gbrtMod
    if model == 'RFMod':
        model = RFMod
    if model == 'logRegMod':
        model = logRegMod
    if model == 'adaMod':
        model = adaMod
    if model == 'SVCMod':
        model = SVCMod  

    if not (model and open and high and low and close):
        return page_not_found(404)

    xTest = dataPipeline(opn, high, low, close, 500, 400)
    return jsonify(list(predict(xTest, model)[0]))


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

if __name__ == '__main__':
    app.run(threaded=True)
