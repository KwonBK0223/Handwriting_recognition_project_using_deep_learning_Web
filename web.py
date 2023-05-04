import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests

st.write('# World Master')
st.write('# Prediction of handwritten English character')

@st.cache(allow_output_mutation=True)
def load():
    url = 'https://github.com/KwonBK0223/streamlit_practice/raw/main/maincnn.h5'
    r = requests.get(url)
    with open('maincnn.h5','wb') as f:
        f.write(r.content)        
    model = load_model('maincnn.h5')
    return model
model = load()

# 알파벳 대문자 레이블
labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# 홈 페이지
def home():
    CANVAS_SIZE = 192

    col1, col2 = st.columns(2)

    with col1:
        canvas = st_canvas(
            fill_color='#000000',
            stroke_width=12,
            stroke_color='#FFFFFF',
            background_color='#000000',
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            drawing_mode='freedraw',
            key='canvas'
        )

    if canvas.image_data is not None:
        img = canvas.image_data.astype(np.uint8)
        img = cv2.resize(img, dsize=(28, 28))
        preview_img = cv2.resize(img, dsize=(CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)

        col2.image(preview_img)

        x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = x.reshape((-1, 28, 28, 1))
        y = model.predict(x).squeeze()

    # 예측 결과의 최댓값 인덱스를 구함
    pred_idx = np.argmax(y)
    # 레이블에 해당하는 문자를 가져옴
    pred_char = labels[pred_idx]
    
    st.write('## Prediction : %s' % pred_char)
    st.bar_chart(y)

# 두 번째 페이지
def page1():
    st.write("What is CNN?")

# 세 번째 페이지
def page2():
    st.write("Code")

# 네 번째 페이지
def page3():
    st.write("# Team mate")

# 메뉴 생성
menu = ['Prediction', 'What is CNN', 'Code']
choice = st.selectbox("메뉴", menu)

# 메뉴에 따른 페이지 선택
if choice == 'Prediction':
    home()
elif choice == 'What is CNN':
    page1()
elif choice == 'Code':
    page2()
elif choice == 'Team Mate':
    page3()

