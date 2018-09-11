# Machine Learning Engineer Nanodegree

## Capstone Project: Using Supervised Classification Algorithms to Predict Bank Term Deposit Subscription

In order to promote products and services, financial institutions like banks generally run marketing campaigns using two approaches: 1) mass campaigns, which targets general indiscriminate public, broadcasting the same message to different customers and 2) directed marketing, which targets specific contacts, creating a directing relationship to customers. Banks which run marketing campaigns following the first approach have had their campaigns' performance reduced over time, having less than 1% of positive responses. On the other hand, marketing campaigns which follow the second approach have shown better results compared to the first. For this reason, banks are more likely to spend their budget on directed marketing campaigns than on inefficient mass campaigns. The personal reason to work on this domain background comes from the fact that I've worked on a project related to a bank company with the goal to offer the most coherent products to its customers based on their characteristics. I believe that having applied machine learning techniques could have helped us to get better response rates. The Capstone is a two-staged project. The first is the proposal component, where you can receive valuable feedback about your project idea, design, and proposed solution. This must be completed prior to your implementation and submitting for the capstone project.

## Libs
The project was developed using Python 3.6.3 - Anaconda environment with the libraries:

- Numpy  
- Matplotlib  
- Pandas  
- Seaborn  
- Scikit-Learn
- XGBoost

## Dataset 
The dataset chosen for this project is related to direct marketing campaigns based on phone calls of a Portuguese banking institution. It was obtained by exploring the [**University of California Irvine's Machine Learning Repository**](https://archive.ics.uci.edu/ml/datasets/bank+marketing). The dataset file which will be used is the bank-full.csv and it contains 45211 instances with 17 columns each. The last column is the target label: whether or not the person subscribed a term deposit. The probability for the label 'yes' (did a term deposit) is approximately 12% and for 'no' (didn't do a term deposit) is approximately 88%. The description of the columns follow:

## Bank client features:
- **age**: the age of the client (numeric).
- **job**: the type of job of the client (categorical). Possible values: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'.
- **marital**: the marital status of the client (categorical). Possible values: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed.
- **education**: the education level of the client (categorical). Possible values: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown'.
- **default**: whether or not the client has credit in default (categorical). Possible values: 'no','yes','unknown'.
- **balance**: average yearly balance in Euros (numeric).
- **housing**: whether or not the client has housing loan (categorical). Possible values: 'no','yes','unknown'.
- **loan**: whether or not the client has personal loan (categorical). Possible values: 'no','yes','unknown'.

Features related with the last contact of the current campaign:
- **contact**: contact communication type (categorical). Possible values: 'cellular','telephone'.
- **day**: last contact day of the month (numeric).
- **month**: last contact month of year (categorical). Possible values: 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'.
- **duration**: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). 

Other features:
- **campaign**: number of contacts performed during this campaign and for this client (numeric, includes last contact).
- **pdays**: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted).
- **previous**: number of contacts performed before this campaign and for this client (numeric).
- **poutcome**: outcome of the previous marketing campaign (categorical). Possible values: 'failure','nonexistent','success'.

## Target label:
- **y**: whether or not the client subscribed to a term deposit (categorical). Possible values: 'yes', 'no'.

