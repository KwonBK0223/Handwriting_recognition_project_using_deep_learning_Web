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
    
    st.write("# Modeling with Mathematics")
    st.write("### Min-Max Scaling & Isomorphic")
    st.write("* Min-Max Scaling")
    st.write("Min-Max Scaling은 입력 데이터의 범위를 0과 1사이로 조정하는 방법이다.")
    st.write("이 스케일링을 수행하면 입력 데이터의 최솟값은 0, 최댓값은 1이 되도록 조정된다.")
    st.write("이러한 스케일링은 입력 데이터의 모든 값이 일정한 비율로 조정되므로 데이터 간의 상대적인 크기가 보존된다.")
    st.write("* Isomorphic")
    st.write("Isomorphic은 두 개의 대상 A와 B가 있을 때, A와 B 간의 매핑 f가 존재하고,")
    st.write("이 매핑이 일대일 대응이면서 연산을 보존하며 역함수가 존재하는 경우를 의미합니다.")
    st.write("이는 대상 A와 B 간의 구조와 내부 연산이 서로 동일하게 보존되는 것을 의미합니다.")
    st.write("* Min-Max Scaling & Isomorphic")
    st.write("Min-Max 스케일링은 입력 데이터에 대한 일대일 대응 매핑을 수행하고, 이로 인해 데이터 간의 상대적인 크기가 보존되므로 Isomorphic 이다.")
    
    st.write("### What is ReLU")
    st.write("* ReLU(Rectified Linear Unit)")
    st.latex(r''' f(x) = max(0,x)''')
    st.write("입력값이 양수일 경우 그대로 출력하고, 음수일 경우 0으로 출력하는 함수")
    st.write("* Why use ReLU")
    st.write("1. 비선형 함수")
    st.write("ReLU 함수는 비선형 함수로서, 딥러닝 모델에서 비선형성을 추가할 때 많이 사용됩니다.")
    st.write("비선형 함수를 사용함으로써 모델이 다양한 패턴과 특징을 학습할 수 있게 됩니다.")
    st.write("2. Gradient vanishing 문제 해결")
    st.write("Sigmoid나 Tanh와 같은 함수는 입력값의 절댓값이 커질수록 기울기가 0에 가까워지는 gradient vanishing 문제를 가지고 있습니다.")
    st.write("하지만 ReLU 함수는 입력값이 음수일 때 기울기가 0이기 때문에 gradient vanishing 문제를 해결할 수 있습니다.")
    st.write("3. 계산 속도 향상")
    st.write("ReLU 함수는 계산 속도가 빠르기 때문에 딥러닝 모델의 학습 속도를 향상시킬 수 있습니다.")
    st.write("4. Sparsity")
    st.write("입력값이 음수일 경우 출력값이 0이 되기 때문에, ReLU 함수를 사용하는 모델은 자연스럽게 sparse한 모델이 됩니다.")
    st.write("이는 모델이 더 간단해지고, 과적합을 방지할 수 있습니다.")
    
    st.write("### What is Softmax")
    st.write("* Softmax")
    st.latex(r'''softmax(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}''')
    st.write("분류 문제에서 출력층의 활성화 값을 확률로 변환해주는 함수")
    st.write("출력층에서 다중 클래스 분류문제에 사용되며, 모든 클래스에 대한 예측값의 합이 1이 되도록 만들어준다.")
    st.write("이를 통해 각 클래스에 대한 확률 값을 계산할 수 있다.")
    st.write("* Why use sofrmax")
    st.write("1. 출력값이 확률로 나타낼 수 있어 해석하기 쉽습니다.")
    st.write("2. 분류 모델에서 모델의 예측값과 정답값 사이의 차이를 계산하는 손실 함수인 cross-entropy와 함께 사용하면, 효과적인 학습이 가능합니다.")
    
    st.write("### What is categorical cross-entropy loss")
    st.write("* categorical cross-entropy loss")
    st.latex(r'''loss = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^M y_{ij} \log(p_{ij})''')
    st.write(" 다중 클래스 분류에서 사용되는 손실 함수")
    st.write("분류 문제에서 예측값과 실제값 사이의 오차를 계산하는 데 사용됩니다.")
    st.write("cross-entropy loss는 확률 분포 사이의 차이를 측정하며, categorical cross-entropy loss는 다중 클래스 분류 문제에 적합합니다.")
    st.write("Why use categorical cross-entropy loss")
    st.write("1. 그래디언트가 0이 아니기 때문에 모든 레이어에서 학습이 가능하다.")
    st.write("2. 확률 분포를 사용하기 때문에 모델이 확률적인 예측을 하게된다.")

    st.write("### Optimizer with Adam")
    st.write("* Optimizer")
    st.write("최적화 알고리즘은 모델이 학습하는 과정에서 최적의 가중치(weight)와 편향(bias)을 찾아내기 위한 방법을 제공합니다.")
    st.write("이 과정에서 손실 함수(loss function)를 최소화하기 위한 가중치와 편향을 찾는 것이 목적입니다.")
    st.write("* Adam")
    st.latex(r'''
    m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \\
    v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \\
    \hat{m}_t = \frac{m_t}{1-\beta_1^t} \\
    \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
    \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t
    ''')
    st.write(r"$w$: 학습되는 가중치")
    st.write(r"$g_t$: 현재 그래디언트")
    st.write(r"$m_t$: 이전 시간 단계의 지수 가중 이동 평균")
    st.write(r"$v_t$: 이전 시간 단계의 제곱 그래디언트의 지수 가중 이동 평균")
    st.write(r"$\beta_1$: 첫 번째 모멘트의 지수적 감소율")
    st.write(r"$\beta_2$: 두 번째 모멘트의 지수적 감소율")
    st.write(r"$\eta$: 학습률")
    st.write(r"$\epsilon$: 분모를 0으로 나누는 것을 방지하기 위한 작은 값")
    st.write("")
    st.write("")
    st.write("경사하강법(gradient descent)의 변형 알고리즘입니다.")
    st.write("Adam은 각 가중치에 대한 적응형 학습률(adaptive learning rate)을 사용해 효율적인 학습을 가능하게 합니다. ")
    st.write("* Why use Adam")
    st.write("1. 학습률을 동적으로 조절하여 학습 속도를 높이고, 과적합(overfitting)을 방지하며, 성능을 최적화하는 효과를 가져옵니다.")
    st.write("2. Adam은 다른 최적화 알고리즘들과 비교했을 때 빠르게 수렴하고, 대부분의 경우에서 성능이 우수하다는 장점이 있습니다.")
    
    
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
    
    st.write("### Library Version")
    st.write("* Python version: 3.9.12")
    st.write("* pandas version: 1.4.2")
    st.write("* numpy version: 1.21.5")
    st.write("* matplotlib version: 3.5.1")
    st.write("* seaborn version: 0.11.2")
    st.write("* tensorflow version: 2.11.0")
    st.write("* cv2 version: 4.7.0")
    st.write("* sklearn version: 1.0.2")
    st.write("* keras version: 2.10.0")

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
menu = ['Prediction', 'What is CNN & Modeling with Mathematics', 'Model Summary','Reflections & Improvements','Team Mate']
choice = st.selectbox("Menu", menu)

# 메뉴에 따른 페이지 선택
if choice == 'Prediction':
    home()
elif choice == 'What is CNN & Modeling with Mathematics':
    CNN()
elif choice == 'Model Summary':
    model_summary()
elif choice == 'Reflections & Improvements':
    review()
elif choice == 'Team Mate':
    Team_Mate()
