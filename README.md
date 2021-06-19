# 이미지분류

# 목차

- [프로젝트 소개](#프로젝트-소개)
- [Problem](#problem)
  - [회고록](#회고록)

## 프로젝트 소개

### 이미지 분류

- 코로나로 인해 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다.
- 주어진 사람의 이미지를 이용해서 마스크 착용여부를 판단합니다.

- input : 마스크 착용 사진, 미착용 사진, 혹은 이상하게 착용한 사진(코스크, 턱스크)
- output : 18개의 class중 하나
마스크 착용여부, 성별, 나이를 기준으로 총 18개의 class 있습니다.

<img src = "https://user-images.githubusercontent.com/59329586/122631415-aac55480-d106-11eb-99f3-b67606f47cbc.png" width="70%" height="35%">

### 평가방법

- 모델은 **F1-Score** 로 평가됩니다.

## Problem

- 많은 label

분류해야하는 label은 18개였기 때문에 데이터에 비해 label 많다고 생각했습니다.

모델을 mask착용유무, 성별구분, 나이구분 3개를 구현했습니다. 이를 종합해서 label분류하는 방식으로 성능을 개선할 수 있었습니다.


- 부족한 데이터

주어진 사람은 4500명이고 학습에 이용한 데이터는 60%인 2700명이었습니다. 그리고 한 명당 이미지갯수는 7개로 총 학습 데이터는 18900장이었습니다.

이를 두 개의 이미지를 랜덤하게 섞는 방식(mix up)으로 해결했습니다. 정상적인 이미지, mix up한 이미지를 번갈아가면서 학습을 했고 이에 맞게 loss를 재정의 해서 성능을 개선할 수 있었습니다.

### [회고록](https://www.notion.so/69decf96ae41409a997b985d3f0c60d5)

## usage

```python
## train
python train.py

## train mask
python train_mask.py

## inference
python inference.py

## inference total
python inference_total.py
```
