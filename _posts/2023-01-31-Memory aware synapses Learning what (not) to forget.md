---
title : Memory aware synapses Learning what (not) to forget (2018)
tags : 논문리뷰 
---

# Memory Aware Synapses : Learning what (not) to forget (ECCV, 2018)

## Background 
### Catastrophic forgetting 
Neural Network가 다른 종류의 task를 학습하면 이전에 학습했던 task에 대한 성능이 감소하는 현성을 의미함 

![](/assets/img/2023-01-31-14-55-03.png)
<br/> 

### Continual Learning 
Catastrophic forgetting을 해결하기 위해 나온 알고리즘으로 하나의 모델을 조금씩 업그레이드 시키면서, 여러 task를 처리할 수 있도록 만드는 방법임. <br/>

![](/assets/img/2023-01-31-15-00-25.png)
<br/> 

## Introduction 
### Motivation 
기존의 Continual Learning은 두 가지 task 사이에서 수행되거나, 여유로운 model의 capacity에서 수행되었음. 반면에 real world에서는 제한된 capacity에서 여러가지 task가 예속해서 주어지는 상황을 수행해야 함. 이전 task들에 대해 model이 전부 기억하려고 하기 보다는, 덜 중요한 정보는 잊고 중요한 정보는 보존하는 방식을 제안함. 즉, network parameter weight들의 importance를 계산하고, 계산된 importance를 기반으로 regularization term을 통해 중요한 weight의 업데이트를 방지함. <br/>
![](/assets/img/2023-01-31-15-03-13.png)
<br/>

## Proposed Method 
### Importance Weight 
Importance Weight는 network의 parameter 변화에 대한 learned function의 sensitivity를 의미한. Data point에 대해서 parameter가 small pertubation으로 변했을 때, output function의 변화량을 아래 수식과 같이 근사함. 
<br/>
![](/assets/img/2023-01-31-15-04-58.png)
<br/>

그리고 Importance weight는 모든 data point에서 parameter에 대한 output function 변화량의 평균으로 아래 수식과 같이 오메가_ij로 나타낼 수 있음. 
<br/>
![](/assets/img/2023-01-31-15-07-05.png)

<br/>

#### L2 Norm 
Neural Network의 output function이 multi-dimension인 경우 계산해야 하는 gradient가 많아짐. Efficient Alternative로, L2 norm을 취한 learned function output의 gradient를 아래와 같이 계산할 수 있음
<br/>
![](/assets/img/2023-01-31-15-08-23.png)
<br/>

L2 norm과 vector form은 실험적으로 significant한 차이가 없으나, L2 norm을 사용했을 때 n(Length of output vector)배 빠르게 계산을 수행할 수 있음 

### Regularization
N번째 task의 학습에 대해 loss function은 아래 수식과 같이 정의될 수 있음 
<br/>
![](/assets/img/2023-01-31-15-09-44.png)
<br/>

Important weight가 크면 새로운 파라미터가 기존 parameter에서 크게 변하지 못하고, Importance weight가 작으면 새로운 파라미터가 기존 파라미터에서 크게 변화함 

### Local & Global MAS 
Global MAS와는 다르게 Local MAS는 전체 네트워크가 아닌 각 layer의 output function F에 근사함. 
<br/>
![](/assets/img/2023-01-31-15-12-45.png)
<br/>

하나의 F가 아니라 전체 F가 각 layer별로 Function의 합성이라고 본 관점임. 아래와 같은 수식을 통해 importance weight를 계산할 수 있음 
<br/>
![](/assets/img/2023-01-31-15-13-51.png)
<br/>

## Experiment
### Performance comparision of MAS
아래 성능 표는 ImageNet으로 pretrain한 AlexNet을 기반으로 실험을 수행한 결과이며  MIT scene, Caltech-UCSD, Oxford Flowers 데이터에 대해 실험을 수행하였음. 서로 다른 task에 대한 training은 SGD, batch size는 200, 100번의 반복을 실험 조건으로 세팅함. 
<br/>
성능 표에서의 괄호()는 _를 의미함 <br/>
![](/assets/img/2023-01-31-15-14-22.png)
<br/>

아래 성능 표는 Global MAS와 Local MAS의 성능을 비교한 표임. MAS와 l-MAS 모두 이전 task에서 performance drop이 크지 않은 것을 확인할 수 있음.<br/>

![](/assets/img/2023-01-31-15-16-21.png)
<br/>

8가지 sequence task에 대해 실험을 수행한 결과는 아래와 같음.  Stanford Cars, FGVC-Aircraft, VOC Actions, Letters, SVHN, 5가지 Task dataset을 사용했고, Flower → Scenes → Birds → Cars → Aircraft → Actions → Letters → SVHN 순으로 학습함 <br/>

![](/assets/img/2023-01-31-15-16-39.png)
<br/>



