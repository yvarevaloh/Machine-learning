from flask import Flask, request
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

import numpy as np
from flask import jsonify
from flask import request
import _pickle as cPickle
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from tablib import Dataset
import json
import os
import numpy as np
import re
import random
import csv

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import tensorflow as tf
import tensorflow_hub as hub
import keras
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Lambda, Dropout, BatchNormalization

app = Flask(__name__)

@app.before_first_request

def init():
    with tf.keras.backend.get_session().graph.as_default():
        embed_size = 512
        category_counts=104
        input_text = Input(shape=(1,), dtype=tf.string)
        embedding = Lambda(UniversalEmbedding, output_shape=(embed_size,))(input_text)
        dense_1 = Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(embedding)
        dense_2 = Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(dense_1)
        dense_3 = Dense(128, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(dense_2)
        pred = Dense(category_counts, activation='softmax')(dense_3)

        return (Model(inputs=[input_text], outputs=pred))

def pronos_position(predicts):
    pos0=[]
    pos1=[]
    pos2=[]
    proba0=[]
    proba1=[]
    proba2=[]
    for k in range(len(predicts)):
        pronos=pd.DataFrame({'position':np.arange(0,len(predicts[k])),'proba':list(predicts[k])}).sort_values(by=['proba'],ascending=False)
        pos0.append(list(pronos['position'])[0])
        pos1.append(list(pronos['position'])[1])
        pos2.append(list(pronos['position'])[2])
        proba0.append(list(pronos['proba'])[0])
        proba1.append(list(pronos['proba'])[1])
        proba2.append(list(pronos['proba'])[2])
    return (pos0,pos1,pos2,proba0,proba1,proba2) 

def build_dict(x,EY_info):
    #print(x)
    predict_var=[x['predicted_account1'],x['predicted_account2'],x['predicted_account3']]
    predict_proba=[x['proba_account1'],x['proba_account2'],x['proba_account3']]
    # matches.append({"eyCode":filter_data['EY Account'].iloc[i],"eyDescription":filter_data['EY #Description'].iloc[i],"score":round(predict_proba[j],2)}) 
    matches=''
    for j in range(3):

        filter_data=EY_info[EY_info['EY Description']==predict_var[j]]
        
        for i in range(filter_data.shape[0]):
            
            if i==0:
                matches=matches+','+'{'+'"eyCode":{eyCode},"eyDescription":"{eyDescription}","score":{score}'.format(eyCode=filter_data['EY Account'].iloc[i],eyDescription=filter_data['EY Description'].iloc[i],score=round(predict_proba[j],2))+'}' 
            else:
                matches=matches+','+'{'+'"eyCode":{eyCode},"eyDescription":"{eyDescription}","score":{score}'.format(eyCode=filter_data['EY Account'].iloc[i],eyDescription=filter_data['EY Description'].iloc[i],score=round(predict_proba[j],2))+'}'             
    
    return('{'+'"account_code_client":{code_account},"account_client":"{cliente_desc}","matches":[{matchesEY}]'.format(code_account=x['code_customer'],cliente_desc=x['account_customer'],matchesEY=matches[1:len(matches)])+'}') 

def UniversalEmbedding(x):
    import tensorflow_hub as hub
    import tensorflow as tf
    embed = hub.Module('../sentence_wise_email/module/module_useT', trainable=True)
    return embed(tf.squeeze(tf.cast(x, tf.string)),signature="default", as_dict=True)["default"]

def exceltodict(x):
    excel_dict={}
    for l in x.headers:
        excel_dict.update({str(l):list(x[l])})
    return excel_dict  

@app.route('/post', methods=['POST'])
                                                                         
def post_route():
    
    if request.method == 'POST':

        file= request.files['file'].read()
        dataset = Dataset().load(file)
        data=exceltodict(dataset)

        formatoEYcomp=pd.read_excel('../sources/formatoEYcomp.xlsx')
        X=data['G/L Description']
        test_text = np.array(X, dtype=object)[:, np.newaxis]
        model=init()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        with tf.keras.backend.get_session().graph.as_default():
            with tf.Session() as session:
                tf.keras.backend.set_session(session)
                session.run(tf.global_variables_initializer())
                session.run(tf.tables_initializer())
                #model.load_weights('./model_5_layersv2.h5')
		model.load_weights('model_5_layersv2.h5')
                predicts = model.predict(test_text, batch_size=32)
        
        with open('../sources/EY_description.csv','rt')as f:
            res=[]
            rd = csv.reader(f)
            for row in rd:
                res.append(row[0])        
        categories=sorted(set(res))
        #predict_logits = predicts.argmax(axis=1)
        predict_logits1,predict_logits2,predict_logits3,proba1,proba2,proba3 = pronos_position(predicts)
        predict_labels1 = [categories[predict_logits1[l]] for l in range(len(predict_logits1))]
        predict_labels2 = ['' if proba1[l]>0.9 else categories[predict_logits2[l]] for l in range(len(predict_logits2))]
        predict_labels3 = ['' if proba1[l]>0.9 else categories[predict_logits3[l]] for l in range(len(predict_logits3))]
        predicted_data=pd.DataFrame({'code_customer':data['G/L Account'] ,'account_customer':list(X),
                                    'predicted_account1':predict_labels1,
                                     'predicted_account2':predict_labels2,'predicted_account3':predict_labels3,
                                     'proba_account1':proba1,'proba_account2':proba2,'proba_account3':proba3})
        
        EY_info=formatoEYcomp[['EY Account','EY Description']].drop_duplicates().sort_values(by=['EY Description'])    
        build_dict_com=''
        for k in range(predicted_data.shape[0]):
            x=predicted_data.iloc[k]
            if k==0:
                build_dict_com=build_dict_com+build_dict(x,EY_info)
            else:
                build_dict_com=build_dict_com+','+build_dict(x,EY_info)
                
  
        return jsonify('{'+'"accounts":[{list_result}]'.format(list_result=build_dict_com)+'}')


app.run()