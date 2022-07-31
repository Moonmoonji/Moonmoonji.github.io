---
title : <논문리뷰> Recurrent Reconstructive Network for Sequential Anomaly Detection
tags : 논문리뷰 
---

# Introduction 
일반적으로 데이터는 training을 위한 적절한 라벨 데이터가 없는 경우가 많다. 따라서 학습된 representation이 downstream 작어에 사용될 수 있는 시계열에 대한 unsupervised representation learning에 대해 연구하는 것이 중요하다. Unsupervised Representation learning은 컴퓨터 비전과 자연어처리 분야에서 많이 연구되었지만 시계열 분석에는 아직 많이 연구되어 있지 않다. 
<br/>

최신 연구는 주로 contrastive learning fremework를 시계열 데이터에 대한 비지도 표현학습에 사용한다. 선행연구로는 아래와 같은 method가 존재한다. 
<br/> 
  
* SRL(Scalable Representation Learning) : triplet loss 개념 도입 
* CPC(Contrastive Predictive Coding) : 손실함수에 대한 Noise-Contrastive Estimation에 의존하여 잠재공간에서 강력한 자기 회귀 모델을 사용하여 미래에 예측을 수행하므로써 표현학습 수행 
* TS-TCC(Temporal and Contextual Contrasting) : CPC의 개선된 작업이면 다른 timestamp와 augmentation에 의해 도입된 perturbation(혼란)에 대해 더 어려운 예측 작업을 통해 강력한 표현을 학습 
* TNC(Temporal Neighborhood Coding) : 새로운 이웃 기반 비지도 학습 프레임워크를 제시하고 비정상(non-stationary) 다변량 시계열에 대한 샘플 가중치 조정을 적용함 
  
이들의 주요 차이점은 시간 슬라이싱을 기반으로 서로 다른 샘플링 정책에 따라 대조적인 쌍을 선택한다는 것이다. 그러나 그러한 정책은 fale negatives의 영향을 받기 쉬우며 global semantical information 손실로 인해 장기적인 종속성을 포착하지 못한다. 게다가 그들은 시간적 특징만 추출하고 Spectral 특징의 활용을 무시하며 성능에 영향을 미칠 수 있는 temporal-spectral relation을 포함한다. 

