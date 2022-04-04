# Coronavirus-Disease-Prediction

This project aims to predict people who will Coronavirus disease using machine learning models with the help of clinical data of the patients.


# What is  Coronavirus Disease?

Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. The COVID-19 pandemic is a formidable global public health challenge. Since the initial emergence of a novel coronavirus in late 2019, the spread of SARS-CoV-2 has been unrelenting, impacting nearly every aspect of society worldwide.

# Dataset

Two separate datasets were studied. Dataset 1 contains 29 clinical data for 280 patients. Dataset 2 contains 28 clinical data for 36 patients. Data were obtained from the hospital. Therefore, it will increase. (I will add it here as it changes)

# Data Preprocessing

In the preprocessing part, first I cleaned the data and according to skew-distribution fill the missing values of them. After that, I reduced the values to the range 0-1 with normalization.Then, I applied Label Encoding, One-Hot Encoding and Multi Label Binarizer to the categorical clinic data .

## Normalization:

The following  features are  data encoded with the normalization method.

 - yaş
 - yattığı gün sayısı
 - saturasyon
 - nörofil plazma öncesi
 - lenfosit plazma öncesi
 - trombosit plazma öncesi
 - crp plazma öncesi
 - ürik asit plazma öncesi
 - d dimer plazma öncesi
 - fibrinojen plazma öncesi
 - troponin plazma öncesi
 - prokalsionin plazma öncesi
 - ferriin plazma öncesi
 - lenfosit son
 - plt son
 - crp son
 - ferritin son
 - fibrinojen son
 - lökosit son
 - prokalsionin son
 - trop son
 - d dimer son
 
## Label Encoding:

The following categorical features are ordinal data encoded with the label encoding method.

 - cinsiyet
 - pcr
 - ht
 - dm
 - malignite
 - tiroit
 - koah
 - astım
 - kky
 - kby
 - gebelik
 - prostat
 - nörolojik
 - kah
 

## One-Hot Encoding:

The following categorical features are nominal data encoded with the one-hot encoding method.

 - Ek hastalıklar
 - Sonuç

## Multi-Label Binarizer:

The following categorical features are ordinal data encoded with the multi label binarizer method.

 - ek hastalıklar
	 - **How was it used :**  Diseases are given numerically as 1,2,3,....14.
		I converted these numbers to binary system using multi label.
		For example; The value of the patient with 1 disease; It has become 01000000000000.


## Train-Test Split:

The data is divided into three as train, test and validation.

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)  
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.25)


## Model Building:

Many models have been created.

 -  Cnn
 - Decision tree
 - Used cross validation
 


>  **The project is still on going. I'll add it here as the project is updated...**
