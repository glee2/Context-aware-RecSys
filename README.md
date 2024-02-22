# Context-aware-RecSys
Development of a context-aware serendipitous recommender system

### 프로젝트 배경 및 필요성
- 콘텐츠 소비자들은 뻔하지 않으면서도 취향에 부합하는 추천을 받을 때 큰 만족감을 느낌
- 추천 시스템의 정확성과 다양성은 trade-off 관계를 가짐
- 예상치 못한 만족을 주는(serendipitous) 추천을 위해서는 이러한 trade-off를 최소화하는 추천 시스템의 개발이 필요함

![image](https://github.com/glee2/Context-aware-RecSys/assets/18283601/deb47bd2-040e-4dd4-836d-37742e51604c)

### 프로젝트 목표
- 사용자의 콘텐츠 소비 맥락을 파악하고, 이를 기반으로 사용자의 취향에 맞으면서 도 다양한 맥락을 제공해 예상치 못한 만 족을 주는 영화 추천 시스템을 개발하는 것을 목표로 함

### 데이터 - MovieLens 영화 평점 데이터
- 영화 소비 패턴을 추출하기 위해 사용자별 영화 평점 기록 정보를 활용
- 영화 평점이 기록된 순서를 바탕으로 추천의 정확성 측정
- 사용자의 영화 시청 순서 사이의 맥락 변화를 바탕으로 추천의 다양성 측정

![image](https://github.com/glee2/Context-aware-RecSys/assets/18283601/3acbb174-ec16-4c5a-9ccf-3706156e397a)

### 분석 방법 – LDA (Latent Dirichlet allocation)
- 사용자별 영화 평점 데이터에 대해 LDA를 수행하여, 평점이 기록된 영화의 순서와 빈도를 기반으로 6개의 대표 토픽을 추출하고, 각각의 토픽을 영화를 시청하는 하나의 맥락으로 정의
- 영화를 시청하는 맥락이 장르뿐 아니라 분위기, 연령대, 연관 감독 및 배우 등 다양한 형태로 나타남

![image](https://github.com/glee2/Context-aware-RecSys/assets/18283601/f66a8445-a750-4366-9bc3-af0a1c419222)


### 분석 방법 – Word2Vec
- 영화 평점 데이터를 학습과 테스트셋으로 나누어, 학습셋에 대해 Word2Vec 기법을 적용하여 각 영화를 잠재 공간에 임베딩시킴
- Word2Vec 학습은 Bagging 앙상블 방식으로 이루어져, 학습셋에 포함된 사용자들의 맥락에 과적합되는 현상을 방지
- 테스트셋의 각 사용자별 평점 기록의 앞부분은 Query 시퀀스로, 뒷부분은 Test 시퀀스로 분리함
- Query 시퀀스의 영화들에 대한 임베딩 벡터를 모두 더하여 최종 벡터를 산출하고, 잠재 공간 내 가까운 순으로 배치된 영화들을 추천 항목으로 반환함

![image](https://github.com/glee2/Context-aware-RecSys/assets/18283601/c727780f-69e3-4863-beb3-350f7c918c9c)

### 추천의 정확성 및 다양성 측정
- 정확성(Accuracy)은 Test 시퀀스 내에 모델이 추천한 항목이 얼마나 포함되는지 Hit ratio로 계산됨

![image](https://github.com/glee2/Context-aware-RecSys/assets/18283601/a823c49a-cf74-4e14-811a-29cf4e6fb5e6)

- 다양성(Diversity)은 Query 시퀀스에 포함된 영화들의 맥락과, 모델이 추천한 항목들의 맥락을 비교하여 기존 시청 이력에 비해 얼마나 새로운 맥락의 영화가 추천되었는지 비율로 계산됨

![image](https://github.com/glee2/Context-aware-RecSys/assets/18283601/ae270a0e-b186-4def-b190-2accb2f6210b)

### 성능 검증 실험 결과
- 추천 성능 검증을 위해 일반적인 협업 필터링(CF) 기법과, 랜덤 추천(RD) 방식의 추천 결과와 비교함
- 다른 방식에 비해 추천의 정확성과 다양성의 trade-off 관계가 개선되었음
- Word2Vec 방식의 경우, 정확성과 다양성 값 사이에 특정한 경향이 없이 flat한 형태를 보임 → trade-off 완화

![image](https://github.com/glee2/Context-aware-RecSys/assets/18283601/204ca25f-6e76-450e-bb16-a3907a65540d)
