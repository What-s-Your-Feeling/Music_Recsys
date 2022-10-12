# 감성분석에 따른 음악 추천 알고리즘

## Architecture Brief
<img src="https://user-images.githubusercontent.com/54973366/186615945-31aa4f87-ca3a-41d1-9068-9a7d6a0d2021.svg" width="800" height="667"/>

## Emotion_classification
- 입력 값으로 받은 사용자의 문장을 6개의 라벨(분노/긴장/슬픔/평화/기쁨/중립)을 기준으로 Multi-class sentiment classification을 통해 감정을 분류합니다.  

- 학습에 사용되는 데이터는 AI-HUB 감성대화 말뭉치와 한국어 감정 정보가 포함된 단발성 대화 데이터셋입니다.  

- 모델 선정 과정은 다음과 같습니다.

|모델|F1_score|Precision|Recall|
|------|---|---|---|
|KcBert|.714|.717|.719|
|Klue-RoBERTa-large|.723|.722|.726|
|Kc-Electra|.713|.713|.715|
|KoBERT|.629|.625|.616|

최종적으로 Klue-RoBETa-large로 결정되었습니다.

성능 향상을 위해 AI-HUB 일상어 데이터를 INPUT으로 DAPT기법을 적용하였습니다. 도메인의 범위가 큰 순서대로 PRE-TRAIN의 기준으로 삼고 진행하였습니다.(가사-일상어-감성어)

모델 하이퍼파라미터 튜닝의 경우 lr,optimizer,patience 등을 수정하면서 실험을 진행하였으며 결과는 아래 그래프와 같습니다.
|LOSS|F1_score|
|------|---|
|![](/readme_image/wandb_graph1.png)|![](/readme_image/wandb_graph2.png)|


### 선정된 모델의 주요 파라미터 값
|model_name|adafactor_1r1e_6|
|------|---|
Optimizer | adafactor|
batch_size | 32|
learning_rate | 1e-6|



## Lyrics_crawl
- 카카오 아레나(https://arena.kakao.com/c/8) 데이터를 DB를 곡 제목을 기준으로 멜론 웹사이트를 통해 가사를 수집합니다.
- 가사 내용을 기준으로 감성분석을 진행하여 해당 곡이 어떠한 감성(분노/긴장/슬픔/평화/기쁨/중립)에 가까운지 라벨링합니다. 모델은 Emotion_classification와 동일한 모델이 사용됩니다. 

## Mel-Spectrogram
- 카카오 아레나(https://arena.kakao.com/c/8)의 Mel-spectrogram data를 사용하여 멜로디 패턴에서 감성을 찾습니다.
- Mel data를 라벨링하는 과정은 다음과 같습니다.  
    1. 해당 곡의 Mel-spectrogram data는 가사내용의 감성과 동일한 시너지를 낸다고 가정합니다. (가사내용과 상반되는 멜로디가 있을 경우는 배제)
    2. 라벨링된 Mel-spectrogram data를 기준으로 비슷한 형태를 띄는 다른 Mel spectrogram data에 Active learning을 적용하여 라벨링합니다.

- 512차원 임베딩 모델 : https://github.com/music-embedding-aiffelthon/Music-embedding

## Recommendation
- 추천 시스템 작동과정  
    1. 사용자는 지금의 상태나 느낌을 문장으로 표현합니다.
    2. Emotion_classification으로 입력받은 문장에 대해 감성분석을 진행하여 softmax의 마지막 벡터(1,6)를 추출합니다.
    3. Music Database에서 input 문장을 변환한 벡터(1,6)과 가장 가까운 코사인 유사도를 가지는 embedded 벡터를 찾아냅니다.
    
- 추후 발전방향
1. 추천시스템 고도화
    - 사용자 피드백 수집 및 반영
    - 협업 필터링 기능 추가

2. 음성인식 기능 추가
    - 음성인식을 통한 입력 모듈 추가

3. Mel2Text, Text2Mel
    - 멜로디를 주면 가사를 생성하거나, 가사를 주면 멜로디를 생성하는 모델 구현

4. KDT 해커톤 참여
    - 경진대회 참여를 통한 지속 개발
    
