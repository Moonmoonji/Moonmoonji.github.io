---
title : Variational transformer-based anomaly detection approach for multivariate time series(2022)
tags : 논문리뷰 
---

# 논문 리뷰 : Variational transformer-based anomaly detection approach for multivariate time series

## Overview 
### MT-RVAE(Multiscale Transformer-based Variational AutoEncoder)
- Time-Series information과 Period information을 반영한 Positional Encoding 
    - informer에서 처음 제안된 timestamp 정보를 반영한 encoding 
    - Fourier series를 활용한 Periodic encoding (새로 추가됨)

- Multi-scale feature fusion algorithm
    - up-sampling 프로세스에서 누락된 세부 정보를 보충하기 위해 Feature Pyramid Structure 

- Residual variational autoencoder architecture
    - Transformer의 Autoencoder 아키텍처를 Variational Autoencoder 아키텍처로 수정
    - KL divergence vanishing problem 방지를 위해 residual connection 사용 

## Methodology (MT-RVAE)
![](/assets/img/2023-01-16-16-53-55.png)
<br/>

### Vanilla Transformer와의 차이점 
![](/assets/img/2023-01-16-17-04-10.png)
<br/>

1. Multiscale Feature Fusion Moudule을 사용
2. Timesatmp, Period를 고려한 Positional Encoding 
3. Time Attention Module 사용
4. 각 layer의 결과를 Encoder, Decoder의 output과 residual connect
5. Transformer Encoder의 output을 통해 Latent vector Z를 생성하는 VAE구조 

### MT-RVAE의 Encoder Input과 Decoder Input 구성 
- 비지도 학습 기반이므로 전처리 시 anomaly 여부를 활용하지 않음
- 학습 데이터를 기반으로 전체 데이터 정규화 수행 
- 모델 학습 시, 일정 길이의 window를 생성하여 사용 
![](/assets/img/2023-01-16-17-06-08.png)
<br/> 

### Multiscale Feature Fusion Module
- Input Embedding을 위해 보통 fully-connected layer 또는 1D Convolutional layer 사용
- FC layer, 1D conv layer 만으로는 정확한 매핑 관계를 맞출 수 없으므로 Input data의 up-sampling에서 많은 양의 세부 정보가 손실(??) 
- 잘못된 데이터에 대한 모델의 민감도가 감소하여 모델이 시계열 데이터에서 일부 잘못된 데이터를 감지할 수 없게됨 
- up-sampling 프로세스에서 누락된 세부 정보를 보충하기 위해 Feature Pyramid Structure 사용

![](/assets/img/2023-01-16-17-08-46.png)
<br/> 

### Input Representation 
최근 연구들은 positional encoding 말고도 global time stamp를 사용하고 있음. global time stamp를 통해 local한 positional 정보 뿐만 아니라 시간 일과 관련된 global positional 정보 반영 
<br/>

해당 논문은 주기적인 특성도 반영하기위해 periodic encoding까지 사용(fourier 급수 사용). 즉, Time-Series encoding과 Periodic encoding 두 가지를 사용하여 timestamp 정보와 주기 정보를 활용함 

### Residual Variational AutoEncoder 
Transformer 모델을 autoencoder 아키텍처에서 Variational autoencoder 아키텍처로 수정함.
<br/>
이렇게 수정한 이유는 Training data 이외에 normal data를 디코딩할 때 심각한 reconstruction error가 발생하는 것을 방지함 

#### Residual Connection 
- 일반적인 residual 구조와 달리 encoder와 decoder를 연결하지 않고 별도로 결합
- encoder 정보가 decoder로 너무 많이 전달되어 Residual Autoencoder 구조의 실패를 방지 
- 간접적으로 손실함수에서 Loss_res 항의 가중치를 증가 시키고 KL divergence vanishing problem을 방지함 

### Time Attention Module 
Transformer의 sel attention을 통해 feature dimension의 상관관계를 반영할 수 있으나, time dimension의 자기 상관을 고려하지는 않음. Time dimension의 중요도를 추출하기 위해서 Time Attention Module을 사용함. 
<br/>

Decoding 능력이 너무 강해서 KL divergence vanishing problem이 발생하지 않게 하기 위해 해당 모듈을 Encoder에만 사용<br/> 

![](/assets/img/2023-01-16-17-27-28.png)

<br/>

## Experiment
### NAB-MT 데이터셋에 대한 실험 결과 
![](/assets/img/2023-01-16-17-29-50.png) 
<br/>

### SKAB 데이터셋에 대한 실험 결과 
![](/assets/img/2023-01-16-17-32-14.png)

## Reference 
https://www.youtube.com/watch?v=aOkhPG7J7t8 

