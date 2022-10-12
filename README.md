# 감성기반 음악 추천시스템 개발 프로젝트


## Introduction
사용자에게 대화체 또는 구어체 문장을 입력받아 감성을 분석한 뒤 현재 감성과 유사한 음악 5곡을 추천하는 알고리즘입니다.



## Data collection
|Dataset|Description|Source|
|-|-|-|
|온라인 구어체 말뭉치|온라인 플랫폼(게시판, 댓글 등)에서 구어체 위주의 텍스트 데이터|https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=625|
|감성 대화 말뭉치|일반인 1,500명을 대상으로 하여 음성 15,700문장 및 코퍼스 27만 문장 구축|https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86|
|Melon Playlist Continuation|멜론 음악 70만건에 대한 곡정보|https://arena.kakao.com/c/8|

### Lyrics_crawling
- 카카오 아레나(https://arena.kakao.com/c/8) 데이터를 DB를 곡 제목을 기준으로 멜론 웹사이트를 통해 가사를 수집합니다.
- 가사 내용을 기준으로 감성분석을 진행하여 해당 곡이 어떠한 감성(분노/긴장/슬픔/평화/기쁨/중립)에 가까운지 라벨링합니다. 모델은 Emotion_classification와 동일한 모델이 사용됩니다.

### Mel-Spectrogram
- 카카오 아레나(https://arena.kakao.com/c/8)의 Mel-spectrogram data를 사용하여 멜로디 패턴에서 감성을 찾습니다.


- 512차원 임베딩 모델 : https://github.com/music-embedding-aiffelthon/Music-embedding



## Further infomation


### Architecture Brief
<img src="https://user-images.githubusercontent.com/54973366/186615945-31aa4f87-ca3a-41d1-9068-9a7d6a0d2021.svg" width="800" height="667"/>  

KLUE-RoBERTa-Large모델에 가사, 일상어, 감성어를 Masked LM 기반 pre-train을 수행했습니다. Task 데이터셋의 크기가 작고 pre-train 데이터셋과 유사하기 때문에 backbone model을 freeze하여 bottleneck feature를 추출한 뒤 감성어 데이터셋으로 FC layer만 학습시키는 finetuning과정을 거쳤습니다.


### Emotion_classification
- 입력 값으로 받은 사용자의 문장을 6개의 라벨(분노/긴장/슬픔/평화/기쁨/중립)을 기준으로 Multi-class sentiment classification을 통해 감정을 분류합니다.  

- 학습에 사용되는 데이터는 AI-HUB 감성대화 말뭉치와 한국어 감정 정보가 포함된 단발성 대화 데이터셋입니다.  

- 모델 선정 과정은 다음과 같습니다.

|Model|F1_score|Precision|Recall|
|------|---|---|---|
|KcBert|.714|.717|.719|
|Klue-RoBERTa-large|.723|.722|.726|
|Kc-Electra|.713|.713|.715|
|KoBERT|.629|.625|.616|

최종적으로 Klue-RoBETa-large로 결정되었습니다.

성능 향상을 위해 AI-HUB 일상어 데이터를 INPUT으로 DAPT기법을 적용하였습니다. 도메인의 범위가 큰 순서대로 PRE-TRAIN의 기준으로 삼고 진행하였습니다.(가사-일상어-감성어)

모델 하이퍼파라미터 튜닝의 경우 lr,optimizer,patience 등을 수정하면서 실험을 진행하였으며 결과는 아래 그래프와 같습니다.
|Loss|F1_score|
|------|---|
|![](/readme_image/wandb_graph1.png)|![](/readme_image/wandb_graph2.png)|


#### 선정된 모델의 주요 파라미터 값
|Model_name|Adafactor_1r1e_6|
|------|---|
Optimizer | adafactor|
batch_size | 32|
learning_rate | 1e-6|


 

## (main topic) Recommendation
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

Notion : https://www.notion.so/modulabs/2dd3321dfd9c43f2ad7109b2fb165dc8?v=236ebb955f9f4c0790ec8c38539613f5
