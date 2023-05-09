import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# 화면을 최대로 와이드 
st.set_page_config(layout="wide")

# 제목
st.write('# World Master')
st.write('# Prediction of handwritten English character')

# 모델 로드
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

# 메인 예측 페이지
def home():
    CANVAS_SIZE = 192

    # 사용자로부터 숫자 입력 받기
    num_canvas = st.number_input('Enter the number of alphabets you want to enter(1~10)', min_value=1, max_value=10, value=2, step=1)

    # canvas 생성 및 예측 결과 계산
    predictions = ''

    # 5개 단위로 자르기 위해서 줄 나누기
    num_rows = num_canvas // 5 # 몫 => 줄 개수
    num_cols = num_canvas % 5  # 나머지 => 마지막줄
    for row in range(num_rows):
        col_list = st.columns(5)
        for i, col in enumerate(col_list):
            with col:
                canvas = st_canvas(
                    fill_color='#000000',
                    stroke_width=12,
                    stroke_color='#FFFFFF',
                    background_color='#000000',
                    width=CANVAS_SIZE,
                    height=CANVAS_SIZE,
                    drawing_mode='freedraw',
                    key=f'canvas{row}_{i}'  # row 값을 key에 추가
                )

                if canvas.image_data is not None:
                    img = canvas.image_data.astype(np.uint8)
                    img = cv2.resize(img, dsize=(28, 28))

                    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    x = x.reshape((-1, 28, 28, 1))
                    y = model.predict(x).squeeze()

                    # 예측 결과의 최댓값 인덱스를 구함
                    pred_idx = np.argmax(y)
                    # 레이블에 해당하는 문자를 가져옴
                    pred_char = labels[pred_idx]

                    # 예측 결과 문자열에 추가
                    predictions += pred_char
                        
    if num_cols > 0:
        col_list = st.columns(num_cols)
        for i, col in enumerate(col_list):
            with col:
                canvas = st_canvas(
                    fill_color='#000000',
                    stroke_width=12,
                    stroke_color='#FFFFFF',
                    background_color='#000000',
                    width=CANVAS_SIZE,
                    height=CANVAS_SIZE,
                    drawing_mode='freedraw',
                    key=f'canvas{num_rows}_{i}'  # num_rows 값을 key에 추가
                )

                if canvas.image_data is not None:
                    img = canvas.image_data.astype(np.uint8)
                    img = cv2.resize(img, dsize=(28, 28))

                    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    x = x.reshape((-1, 28, 28, 1))
                    y = model.predict(x).squeeze()

                    # 예측 결과의 최댓값 인덱스를 구함
                    pred_idx = np.argmax(y)
                    # 레이블에 해당하는 문자를 가져옴
                    pred_char = labels[pred_idx]

                    # 예측 결과 문자열에 추가
                    predictions += pred_char
    
    # 결과값 출력
    st.write('## Predictions : %s' % predictions)
 
# 개념설명
def CNN():
    st.write("# What is CNN")
    st.write("### CNN 이란")
    st.write("* 기원")
    st.write("데이비드 허블과 토르스텐 비셀이 시각피질 구조 연구(뇌가 이미지를 인식하는 방법을 찾는 연구)를 통해")
    st.write("우리의 뉴런(신경)들이 시야의 일부 범위 안에 있는 시각자극에만 반응한다는 것을 보였다.")
    st.write("즉, 뉴런이 시야의 몇몇 부분에 반응하고 이 부분들을 합쳐서 전체 시야를 감싼다는 것이다.\
이 연구가 CNN, 합성곱 신경망으로 점진적으로 진화되었다.\
")
    st.write("* 구조")
    st.write("<Input layer - Convolution layer - Pooling layer - Fully Connected layer - Ouput layer>")
    st.write("(1) Input layer: 이미지를 입력")
    st.write("(2) Convolution layer: 필터를 통해 이미지에서 특성(feature)을 추출")
    st.write("(3) Pooling layer: 특성맵을 다운샘플링하여 연산량을 감소")
    st.write("(4) Fully Connected layer: 소프트맥스를 활성화함수로 사용하여 다중 분류(Multi-class Classification)")
    st.write("(5) Output layer: 이미지를 분류해 결과 출력")
    
    image_path = 'Image/CNN.PNG'
    img = Image.open(image_path)    
    st.image(img, width = 1000)
    
    st.write("* CNN이 이미지 분류에 많이 사용되는 이유")
    st.write("1. Locality of pixel dependencies")
    st.write("객체를 구성하는 픽셀은 국소적으로만 관련있다.")
    st.write("2. Stationarity of statistics")
    st.write("이미지에는 위치와 관계없이 유사한 패턴이 반복될 수 있는데, 하나의 필터로 이미지를 탐색하며 같은 피쳐를 추출하여 효율적이다.")
    st.write("ex) 입은 사람마다 가지고 있으므로 여러 명이 찍힌 사진에서 입은 컴퓨터 입장에서 유사한 패턴으로 판단할 수 있다.")
    
# 모델 요약
def model_summary():
    st.write("### 1. 입력 데이터와 출력 데이터로 Split")
    image_path = 'Image/1.split.PNG'
    img = Image.open(image_path)    
    st.image(img, width = 1000)
    st.write("* 데이터셋에서 '0' 열을 'label'로 이름을 변경하고, 'label'열을 y로 지정합니다.")
    st.write("* X는 'label'열을 제외한 나머지 열들로 구성합니다.")

    st.write("### 2. 데이터셋 셔플")
    image_path = 'Image/2.shuffle.PNG'
    img = Image.open(image_path)    
    st.image(img, width = 1000)
    st.write("* X 데이터를 셔플합니다.")

    st.write("### 3. train, test set 으로 split & 데이터 값 범위 조정(Scaling)")
    image_path = 'Image/3.scaling.PNG'
    img = Image.open(image_path)    
    st.image(img, width = 1000)
    st.write("* X, y 데이터를 train set과 test set으로 분리합니다.")
    st.write("* MinMaxScaler()를 사용하여 데이터를 스케일링합니다.")
    st.write("* 스케일링한 train set과 test set을 각각 train_scaled, test_scaled에 저장합니다.")
    st.write("* train_scaled에서 첫번째 데이터의 0~9번째 feature값을 출력합니다.")

    st.write("### 4. 데이터 배열의 차원을 변경(Reshape)")
    image_path = 'Image/4.reshape.PNG'
    img = Image.open(image_path)    
    st.image(img, width = 1000)
    st.write("* train_scaled와 test_scaled 데이터를 (샘플수, 가로, 세로, 채널)로 reshape합니다.")

    st.write("### 5. 데이터 형식변환(Convert)")
    image_path = 'Image/5.convert.PNG'
    img = Image.open(image_path)    
    st.image(img, width = 1000)
    st.write("* y_train과 y_test를 one-hot encoding합니다.")
    st.write("* y_train과 y_test의 shape는 (샘플 수, 클래스 수)가 됩니다.")

    st.write("### 6. Modeling")
    image_path = 'Image/6.modeling.PNG'
    img = Image.open(image_path)
    st.image(img, width = 1000)
    st.write("* y_train과 y_test를 one-hot encoding합니다.")
    st.write("* y_train과 y_test의 shape는 (샘플 수, 클래스 수)가 됩니다.")
    st.write("* Sequential() 모델을 생성합니다.")
    st.write("* Conv2D layer를 추가합니다. (filter: 32, kernel_size: (5,5), input_shape: (28,28,1), activation: relu)")
    st.write("* MaxPooling2D layer를 추가합니다. (pool_size: (2,2))")
    st.write("* Dropout layer를 추가합니다. (비율: 0.3)")
    st.write("* Flatten layer를 추가합니다.")
    st.write("* Dense layer를 추가합니다. (unit: 128, activation: relu)")
    st.write("* Dense layer를 추가합니다. (unit: 클래스 수, activation: softmax)")
    st.write("* 모델을 compile합니다. (loss: categorical_crossentropy, optimizer: adam, metrics: accuracy)")
    st.write("* model.summary()를 사용하여 모델 구조를 출력합니다.")

    st.write("### 7. 학습(Fit)")
    image_path = 'Image/7.fit.PNG'
    img = Image.open(image_path)    
    st.image(img, width = 1000)
    st.write("* model.fit() 함수를 사용하여 모델을 학습합니다.")
    st.write("* train_scaled, y_train을 사용하여 학습합니다.")
    st.write("* validation_data에 test_scaled, y_test를 전달하여 validation set으로 모델을 평가합니다.")
    st.write("* epochs는 학습 횟수를 나타내며, 5번의 학습을 수행합니다.")
    st.write("* batch_size는 한 번의 학습에 사용되는 샘플의 수를 나타내며, 200개의 샘플을 한 번의 학습에 사용합니다.")
    st.write("* verbose는 학습 진행 상황을 출력하는 방식을 나타냅니다. verbose=2로 설정하면 epoch마다 학습 손실(loss)과 정확도(accuracy)를 출력합니다.")
    st.write("* 학습이 끝나면, history 변수에 학습 이력을 저장합니다.")
    
    st.write("### 8. 모델 평가")
    image_path = 'Image/8.accuracy.PNG'
    img = Image.open(image_path)    
    st.image(img, width = 1000)
    st.write("* model.evaluate() 함수를 사용하여 test set으로 모델을 평가합니다.")
    st.write("* test_scaled, y_test를 사용하여 모델을 평가합니다.")
    st.write("* verbose=0으로 설정하여 평가 과정을 출력하지 않습니다.")
    st.write("* 모델의 평가 결과인 손실(loss)과 정확도(accuracy)를 출력합니다.")

    st.write("### 9. loss값 시각화(Visualization)")
    image_path = 'Image/9.loss_graph.PNG'
    img = Image.open(image_path)    
    st.image(img, width = 1000)
    st.write("* 모델 학습 과정에서 각 epoch마다 발생한 loss 값을 시각화합니다.")
    st.write("* 그래프를 통해 모델이 학습하는 동안 손실 값이 어떻게 변화하는지, 과적합이 일어나는 지점을 파악할 수 있습니다.")


    st.write("#### Full code")
    st.write("https://github.com/KwonBK0223/Handwriting_recognition_project_using_deep_learning/blob/main/CNN_Modeling/Project_Main_ver.2.0.ipynb")
    st.write("#### 데이터 출처")
    st.write("https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format")
    
# 성찰과 개선점
def review():
    st.write("### 성찰")
    st.write("1. 대문자 알파벳 데이터셋 밖에 구하지 못해서 소문자에는 적용하지 못하였다.")
    st.write("2. 알파벳 철자 이미지를 하나씩 분류하기때문에, 연속적으로 사용한 단어는 인식하지 못한다.")
    st.write("### 개선점")
    st.write("1. 소문자 알파벳 데이터셋을 구한다면 더 다양한 손글씨를 분석할 수 있을 것이다.")
    st.write("2. 연속해서 받은 단어를 알파벳단위로 분할하여 인식할 수 있다면 단어 이미지도 인식 할 수 있을것이다.")
    
# 팀원 페이지
def Team_Mate():
    url = 'https://github.com/KwonBK0223/streamlit_practice/blob/main/Image/PNU_Mark.png?raw=true'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    col1, col2 = st.columns([1,5])
    with col1:
        st.write("\n")
        st.write("\n")
        st.image(img, width = 200)
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.image(img, width = 200)
    with col2:
        st.write("# Leader")
        st.write("## Kwon Byeong Keun")
        st.write("#### PNU Matematics 17 & Industrial Mathematics Software Interdepartmental")
        st.write("#### house9895@naver.com")

        st.write("# Team mate")
        st.write("## Seong Da Som")
        st.write("#### PNU Mathematics 19 & Big Data Interdepartmental")
        st.write("#### som0608@naver.com")
        
# 메뉴 생성
menu = ['Prediction', 'What is CNN', 'Model Summary','Reflections & Improvements','Team Mate']
choice = st.selectbox("Menu", menu)

# 메뉴에 따른 페이지 선택
if choice == 'Prediction':
    home()
elif choice == 'What is CNN':
    CNN()
elif choice == 'Model Summary':
    model_summary()
elif choice == 'Reflections & Improvements':
    review()
elif choice == 'Team Mate':
    Team_Mate()
