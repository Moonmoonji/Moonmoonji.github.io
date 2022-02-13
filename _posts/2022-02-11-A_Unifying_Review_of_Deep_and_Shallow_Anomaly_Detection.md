---
title : [논문 리뷰] A Unifying Review of Deep and Shallow Anomaly Detection(2021)
tags : 논문리뷰 
---

# 논문 리뷰 : A Unifying Review of Deep and Shallow Anomaly Detection 

## What is anomaly?
An anomaly is an observation that deviates considerably from somse concept of normality. 

## Anomaly Detection의 어려움 

1. The variability within normal data can be very large, resulting in misclassifying normal samples as being anomalous (type 1 error) or not identifying the anomalous ones (type 2 error) 
2. Anomalous events are often very rare, which results in highly imbalanced training data sets. Even worse, in most cases, the data set is unlabeled so that it remains unclear which data points are anomalies and why. Hence, AD reduces to an unsuperised learning task with the goal to learn a valid model of the majority of data points. 
3. Anomalies themselves can be very diverse so that it becomes difficult to learn a complete model for them. Likewise, the solution is again to learn a model for the normal samples and treat deviations from it as anomalies. However, this approach can be problematic if the distribution of the normal data changes (nonstationarity), either intrinsically or due to environmental changes (e.g., lighting conditions and recording devices from different manufactures). 

## Type of Anomaly 
### A point anomaly 
### A conditional or contextual anomaly 
### A group or collective anomaly 
