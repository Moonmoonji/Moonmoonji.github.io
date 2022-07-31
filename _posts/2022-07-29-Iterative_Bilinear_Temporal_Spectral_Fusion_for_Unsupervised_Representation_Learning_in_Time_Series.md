---
title : <논문리뷰> Iterative Bilinear Temporal-Spectral Fusion for Unsupervised Representation Learning in Time Series  
tags : 논문리뷰 
---

## Introduction 
일반적으로 데이터는 training을 위한 적절한 라벨 데이터가 없는 경우가 많다. 따라서 학습된 representation이 downstream 작어에 사용될 수 있는 시계열에 대한 unsupervised representation learning에 대해 연구하는 것이 중요하다. Unsupervised Representation learning은 컴퓨터 비전과 자연어처리 분야에서 많이 연구되었지만 시계열 분석에는 아직 많이 연구되어 있지 않다. 
<br/>

최신 연구는 주로 contrastive learning fremework를 시계열 데이터에 대한 비지도 표현학습에 사용한다. 선행연구로는 아래와 같은 method가 존재한다. 
<br/> 
  
* SRL(Scalable Representation Learning) : triplet loss 개념 도입 
* CPC(Contrastive Predictive Coding) : 손실함수에 대한 Noise-Contrastive Estimation에 의존하여 잠재공간에서 강력한 자기 회귀 모델을 사용하여 미래에 예측을 수행하므로써 표현학습 수행 
* TS-TCC(Temporal and Contextual Contrasting) : CPC의 개선된 작업이면 다른 timestamp와 augmentation에 의해 도입된 perturbation(혼란)에 대해 더 어려운 예측 작업을 통해 강력한 표현을 학습 
* TNC(Temporal Neighborhood Coding) : 새로운 이웃 기반 비지도 학습 프레임워크를 제시하고 비정상(non-stationary) 다변량 시계열에 대한 샘플 가중치 조정을 적용함 
  
이들의 주요 차이점은 시간 슬라이싱을 기반으로 서로 다른 샘플링 정책에 따라 대조적인 쌍을 선택한다는 것이다. 그러나 그러한 정책은 fale negatives의 영향을 받기 쉬우며 global semantical information 손실로 인해 장기적인 종속성을 포착하지 못한다. 게다가 그들은 시간적 특징만 추출하고 Spectral 특징의 활용을 무시하며 성능에 영향을 미칠 수 있는 temporal-spectral relation을 포함한다. 

<br/> 

[Figure 1 : Statistics about false predictions on time series classification] <br/> 

![](/assets/img/2022-07-31-15-56-15.png)
<br/>

By spectral 이란 이전 연구에서 제안된 샘플링 방법을 사용하여 대조 쌍을 생성하지만 샘플링된 시계열을 스팩트럼 영역으로 변환하여 나중에 대조 훈련 및 테스트를 위한 특징을 추출하는것을 의미한다. 그림1을 보면 기존 방법 모두 시간적 또는 스팩트럼적 특징이 있는 잘못된 예측에 대해 30%정도 낮은 중복 비율을 가지고 있다. 이는 그들의 temporal, spectral 표현이 각각 독립적으로 학습된다는 것을 증면한다. 

### Contribution 
1. 시계열의 unsupervised representation learning을 위한 기존 연구에서 contrastive pairs의 구성을 다시 검토하고, 전체 시계열을 증가시키기 위해 단순하지만 효과적인 인스턴스 수준 augmentation으로 표준 dropout을 제안하며, 전역 컨텍스트 정보를 최대한 보존하고 기존 시간 슬라이싱 기반 방법을 능가한다. (세그먼트 수준 확대)
2. 새로운 표현 학습 프레임워크 BTSF는 temporal, sepctral 두 도메인의 전역 컨텍스트 정보를 동시에 활용할 뿐만 아니라 새로운 융합-스퀴즈 방식으로 시계열에 대한 교차 도메인 기능 표현을 반복적으로 세분화하는 반복적인 이중 선형 융합으로 쌍별 temporal, spectral 의존성을 명시적으로 모델링하기 위해 제안된다. 
3. 학습된 표현의 일반화 능력을 식별하기 위한 충분한 평가가 수행된다. 분류 외에도 예측 및 이상 탐지와 같은 다른 다운스트림 작업에 대한 모델을 평가하는데, 이는 세가지 작업 모두에 대한 최초의 실험이다. 결과는 bTSF가 기존 모델을 크게 능가할 뿐만 아니라 감독된 기법으로 경쟁력을 갖추고 있음을 보여준다. 

## Method
### Rethinking the construction of contrastive pairs 
![](/assets/img/2022-07-31-19-52-45.png)
<br/>

이전 연구는 training prdcedure에서 contrastive objective를 구성하기 위해 샘플링된 데이터를 사용했다. Sampling bias는 세그먼트 수준 샘플링 정책(시간 슬라이싱)으로 인해 기존 시계열 표현학습의 문제점이다. global semantical information의 손실로 인해 시간 슬라이싱이 장기 종속성을 포착할 수 없다. 
<br/>

global temporal information을 보존하고 시계열에 대한 원래 속성을 변경하지 않기 위해, 우리는 비지도 표현 학습에서 다른 view를 생성하기 위해 표준 드롭아웃을 minimal data augmentation으로 적용한다. 구체적으로,시계열에 독립적으로 샘플링 된 두개의 드롭아웃 마스크를 사용해 양의 쌍을 얻고 다른 변수의 시계열을 음의 쌍 구성을 위한 음의 샘플로 취급한다. Instance-level contrastive pairs를 사용하여 우리의 방법은 장기 종속성을 포착하고 이전 세그먼트 수준 쌍보다 우수한 샘플링 편향을 효과적으로 줄일 수 있다. 

![](/assets/img/2022-07-31-19-53-22.png)
<br/> 
다음과 같은 식으로 x_anc 와 x_pos라는 poitive pair를 얻는다.
<br/>

negative pair를 얻기 위해서는 다변량 시계열에서 랜덤하게 변수를 뽑아서 x_neg를 얻는다. 

### Iterative Bilinear Temporal-Spectral Fusion 
![](/assets/img/2022-07-31-19-59-35.png) 
<br/> 
위의 그림에서 볼 수 있듯이 contrastive pair를 구성한 후에 x와 x_pos를 동화시키고 x_neg와 x_neg를 구별하기 위해 시계열을 고차원 특징 공간에 매핑한다. 이전 연구에서는 스펙트럼 특징과 시간-스펙트럼 관계를 활용하는 것을 무시했는데, 제안하는 BTSF는 스펙트럼과 시간적 특징을 동시에 활용할 뿐만 아니라 표현학습을 보다 세분화된 방식으로 향상시킨다. Summation과 연결 대신, BTSF는 반복적인 이중 선형 시간 스펙트럼 융합을 채택하여 interactive 특징 표현을 생성하기 위한 시간적 특징과 스펙트럼 특징 사이의 쌍별  친화성을 반복적으로 탐색하고 다듬어 positive pairs의 가장 일반적인 부분을 나타내고 negative pairs의 차이를 확대한다. 
<br/> 

구체적으로 각 aumented된 시계열 x_t는 fast Fourier transform을 통해서 spectral domain으로 변환되고 spectral signal x_s를 얻는다. 그리고 x_t와 x_s는 두개의 encoding network로 전달되어 각각 feature를 추출한다. 식은 아래와 같다. 
<br/> 
![](/assets/img/2022-07-31-20-10-54.png) 
<br/> 

Dilated causal convolution의 단순한 stack을 사용해 시간적 특징을 인코딩하고 1D convolutional block을 사용해서 스펙트럼 특징을 추출한다. 그리고 동일한 크기의 feature를 보장하기 위해 인코딩 네트워크 끝에 maxpooling 계층을 적용해 모델 입력 길이로 확장할 수 있다. BTSF는 F_t와 F_s 사이에 반복적인 이중 선형 융합을 한다. 구체적으로 아래와 같이 두 도메인의 기능 간에 채널별 상호 작용을 설정한다. 
<br/> 

![](/assets/img/2022-07-31-20-14-42.png)
<br/> 

여기서 i,j는 각각 temporal, spectral axes의 i번째 j번째 위치이다. BTSF는 획득한 특징  F(i,j)를 통합하여 모든 시간-주파수 특징 쌍의 합으로 초기 이중선형 특징 벡터 F_bilinear를 생성한다. 식은 아래와 같다.
<br/> 
![](/assets/img/2022-07-31-20-18-47.png)
<br/> 

이 biliear feature는 보다 차별적인 피처 표현을 획득하기 위해 세분화된 시간-주파수 친화성을 전달한다. 그런 다음 반복 절차를 통해 시간적 및 스펙트럼 특징을 adaptively하게 개선하기 위해 교차 도메인 친화성을 인코딩한다. 
<br/> 
![](/assets/img/2022-07-31-20-22-56.png)
<br/> 

그럼에도 불구하고, 그것의 효율성은 고차원 특징을 저장하는 메모리 오버헤드로 인해 어려움을 겪을 수 있다. 문제를 해결하기 위해 상호작용 매트릭스 W를 삽입하고 인수분해하여 최종 bilinear feature를 low-rank feature로 변환한다. 먼저 각 시간-스펙트럼 피처 쌍 사이에 선형변환을 만들기 위해서 삽입된다. 
<br/> 
![](/assets/img/2022-07-31-20-25-28.png)
<br/>
![](/assets/img/2022-07-31-20-25-39.png)

<br/> 
각 augmented time series에 대한 Final joint feature representation f는 아래와 같다. 
<br/> 

![](/assets/img/2022-07-31-20-27-03.png) 
<br/> 

또한, 각각의 positive pair의 거리를 최소화 negative pair 거리를 최대화하기 위해 손실함수를 구축한다. 식은 아래와 같다.
<br/> 
 ![](/assets/img/2022-07-31-20-28-49.png) 


## Experiment 
### Classification 
![](/assets/img/2022-07-31-20-31-28.png) 

### Anomaly Detection
![](/assets/img/2022-07-31-20-32-57.png) 
<br/>
SOTA 성능을 달성함 

### Forecasting 
![](/assets/img/2022-07-31-20-32-43.png) 

## Analysis 
### Augmentation comparisons 
인스턴스 수준 augmentation(drop-out) 효과를 입증하기 위해, 12개의 다른 증강 정책과 제안하는 방법을 비교했다. Jittering, Rotation, Scaling, Magnitude Warping, Permuation, Slicing, Time Warping, Window Warping, SPAWNER , Weighted DTW,,,, 등등
<br/>
classification 성능을 비교해보면 제안하는 방법에서 가장 우수했다. 아래 그림 참고
<br/> 
![](/assets/img/2022-07-31-20-43-21.png) 
<br/> 

### Impact of iterative bilinear fusion 
![](/assets/img/2022-07-31-20-48-32.png)
<br/> 

반복적인 bilinear fusion을 추가한 후, BTSF는 정확도에서 큰 프로모션을 차지하고 기존 작업(약 30%)보다 훨씬 높은 96.6%의 중복 비율로 시간영역과 스펙트럼 영역 사이의 양호한 alignment를 달성했다. 따라서 반복적인 이중 선형 융합은 두 도메인 간에 효과적인 상호 작용을 하며 최종 예측 정확도를 위해 필수적임을 확인했다. 

### Alignment and uniformity 
![](/assets/img/2022-07-31-20-51-23.png)
<br/> 

![](/assets/img/2022-07-31-20-51-38.png)
<br/> 

Alignment는 유사한 샘플 간의 특징의 유사성을 측정하는 데 사용되며, 이는 positive pair의 특징이 noise에 의해 변하지 않아야 한다는 것을 의미한다. Uniformity는 잘 학습된 특징 분포가 최대한의 정보를 보존해야 한다고 가정한다. 잘 일반화된 특징 표현은 positive pair의 내부 유사성을 최소화하고 negative pair의 상호 거리를 확대할 뿐만 아니라 충분한 정보를 유지하기 위해 특징을 균일하게 분포되도록 한다. 그림 4와 5는 각각 alignment와 Uniformity의 결과이다. 
