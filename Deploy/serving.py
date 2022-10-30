#-*- coding:utf-8 -*- 
import requests
import pandas as pd
import numpy as np
import streamlit as st
#import streamlit.components.v1 as components

from streamlit_chat import message
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_argument('--no-sandbox')
options.add_argument('--disable-gpu')
options.add_argument('headless')        # 창 x
options.add_argument('--blink-settings=imagesEnabled=false')        # 이미지 x
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.add_experimental_option("excludeSwitches", ['load-extension, enable-automation', 'enable-logging'])
options.add_experimental_option('useAutomationExtension', False)


def youtube_link(artist, song):
    keyword = artist + "+" + song
    temp_url = 'https://www.youtube.com/results?search_query='
    url = temp_url + keyword
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    driver.find_element(By.ID, 'video-title').click()
    header = driver.current_url
    return header




@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained('model', local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained('model', local_files_only=True)
    pipe = pipeline(task = 'text-classification', model = model, tokenizer = tokenizer, return_all_scores = True)
    return pipe


@st.cache(allow_output_mutation=True)
def load_dataset():
    data = pd.read_csv('recsys_data/2015meta.csv')
    return data


query = load_model()
data = load_dataset()

st.title("WYF Music Recommend System")

text = st.text_input('오늘의 기분은 어떤가요?')

label_name = {0:'기쁨',1:'긴장',2:'평화',3:'슬픔',4:'분노',5:'중립'}
def emotion_analysis(input,query):
    emotion,em_list = [],[]
    for i in query(input):
        for j in range(len(i)):
            emotion.append(i[j]['score'])

    for i in np.argsort(emotion)[::-1]:
        em_list.append(label_name[i])
    message(f'당신에게서는 {em_list[:3]}의 감정이 느껴져요')
    message('당신에게 알맞는 노래 분석중 삐-빅')
    st.text('\n')
    return emotion

def cos_sim(vectors,emotion,meta):
    res_list = []
    n= 0
    score = vectors@np.array(emotion) / (((vectors**2).sum(axis=1)**0.5) * ((np.array(emotion)**2).sum()**0.5))
    resys = np.argsort(score)[::-1][:5] #5개 높은 유사도 출력
    message("지금 당신에게 필요한 곡은!")
    st.text("\n")
    st.text("\n")
    st.text("\n")
    st.text("\n")
    for i in resys:
        artist = meta.iloc[i]['artist']
        song = meta.iloc[i]['song_name']
        vec = vectors[i]
        
        res_list.append((artist,song))
        #메인 라벨 3개 출력
        labels = []
        label_num = np.argsort(vec)[::-1]
        for j in label_num[:3]:
            label = label_name[j]
            labels.append(label)
              
        message(f"가수 : {artist},  곡 : {song}")#,    감정 : {labels}")
        st.video(youtube_link(artist, song))



if text:
    message(text, is_user=True)
    with st.spinner():
        emotion = emotion_analysis(text, query)

        vectors = np.load('recsys_data/2015lyric_mel.npy') #lyric+mel 가중합 데이터
        # vectors = np.load('recsys_data/2015lyric.npy')
        # vectors = np.load('recsys_data/2015mel.npy')

        cos_sim(vectors, emotion, data)
