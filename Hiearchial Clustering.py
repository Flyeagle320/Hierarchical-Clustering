# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 12:41:26 2022

@author: Rakesh
"""

import pandas as pd #data manipulation
import numpy as np 
import matplotlib.pyplot as plt
#loading data frame #
airline = pd.read_excel("C:/Users/Rakesh/Downloads/Dataset_Assignment Clustering/EastWestAirlines.xlsx", sheet_name='data')
airline.drop(['ID#'], axis=1 , inplace=True)
airline.columns

#lets check Null and NA values #
airline.isna().sum()
airline.isnull().sum()
#there are no null value#

#checking duplicate value#
dup= airline.duplicated()
sum(dup)
#there is duplicate value lets remove#
airline= airline.drop_duplicates()

airline.columns

##lets find outliers using boxplot#
import seaborn as sns
sns.boxplot(airline.Balance);plt.title('Boxplot');plt.show()
sns.boxplot(airline.Qual_miles);plt.title('Boxplot');plt.show()
sns.boxplot(airline.cc1_miles);plt.title('Boxplot');plt.show()
sns.boxplot(airline.cc2_miles);plt.title('Boxplot');plt.show()
sns.boxplot(airline.cc3_miles);plt.title('Boxplot');plt.show()
sns.boxplot(airline.Bonus_miles);plt.title('Boxplot');plt.show()
sns.boxplot(airline.Bonus_trans);plt.title('Boxplot');plt.show()
sns.boxplot(airline.Flight_miles_12mo);plt.title('Boxplot');plt.show()
sns.boxplot(airline.Flight_trans_12);plt.title('Boxplot');plt.show()
sns.boxplot(airline.Days_since_enroll);plt.title('Boxplot');plt.show()


##scatter plot for Bivariate visualization##
plt.scatter(airline['Balance'],airline['Bonus_miles'])
plt.scatter(airline['Balance'],airline['Bonus_trans'])
plt.scatter(airline['Flight_miles_12mo'],airline['Flight_trans_12'])
plt.scatter(airline['Bonus_miles'],airline['Bonus_trans'])

#lets remove outliers##
IQR = airline['Balance'].quantile(0.75)-airline['Balance'].quantile(0.25)
lower_limit_balance= airline['Balance'].quantile(0.25)-(IQR*1.5)
upper_limit_balance= airline['Balance'].quantile(0.75)+(IQR*1.5)
airline['Balance']=pd.DataFrame(np.where(airline['Balance']>upper_limit_balance,upper_limit_balance,
                                         np.where(airline['Balance']<lower_limit_balance,lower_limit_balance,airline['Balance'])))
sns.boxplot(airline.Balance);plt.title('Boxplot');plt.show()

IQR = airline["Bonus_miles"].quantile(0.75) - airline["Bonus_miles"].quantile(0.25)
lower_limit_Bonus_miles = airline["Bonus_miles"].quantile(0.25) - (IQR * 1.5)
upper_limit_Bonus_miles = airline["Bonus_miles"].quantile(0.75) + (IQR * 1.5)
airline["Bonus_miles"] = pd.DataFrame(np.where(airline["Bonus_miles"] > upper_limit_Bonus_miles , upper_limit_Bonus_miles ,
                                        np.where(airline["Bonus_miles"] < lower_limit_Bonus_miles , lower_limit_Bonus_miles , airline["Bonus_miles"])))
sns.boxplot(airline.Bonus_miles);plt.title('Boxplot');plt.show()

IQR = airline["Bonus_trans"].quantile(0.75) - airline["Bonus_trans"].quantile(0.25)
lower_limit_Bonus_trans = airline["Bonus_trans"].quantile(0.25) - (IQR * 1.5)
upper_limit_Bonus_trans = airline["Bonus_trans"].quantile(0.75) + (IQR * 1.5)
airline["Bonus_trans"] = pd.DataFrame(np.where(airline["Bonus_trans"] > upper_limit_Bonus_trans , upper_limit_Bonus_trans ,
                                    np.where(airline["Bonus_trans"] < lower_limit_Bonus_trans , lower_limit_Bonus_trans , airline["Bonus_trans"])))
sns.boxplot(airline.Bonus_trans);plt.title('Boxplot');plt.show()

IQR = airline['Flight_miles_12mo'].quantile(0.75)-airline['Flight_miles_12mo'].quantile(0.25)
lower_limit_Flight_miles_12mo = airline['Flight_miles_12mo'].quantile(0.25)-(IQR*1.5)
upper_limit_Flight_miles_12mo = airline['Flight_miles_12mo'].quantile(0.75)+(IQR*1.5)
airline['Flight_miles_12mo']=pd.DataFrame(np.where(airline['Flight_miles_12mo']>upper_limit_Flight_miles_12mo,upper_limit_Flight_miles_12mo,
                                                   np.where(airline['Flight_miles_12mo']<lower_limit_Flight_miles_12mo,lower_limit_Flight_miles_12mo,airline['Flight_miles_12mo'])))
sns.boxplot(airline.Flight_miles_12mo);plt.title('Boxplot');plt.show()

IQR = airline['Flight_trans_12'].quantile(0.75)-airline['Flight_trans_12'].quantile(0.25)
lower_limit_Flight_trans_12= airline['Flight_trans_12'].quantile(0.25)-(IQR*1.5)
upper_limit_Flight_trans_12= airline['Flight_trans_12'].quantile(0.75)+(IQR*1.5)
airline['Flight_trans_12']=pd.DataFrame(np.where(airline['Flight_trans_12']>upper_limit_Flight_trans_12,upper_limit_Bonus_trans,
                                                 np.where(airline['Flight_trans_12']<lower_limit_Bonus_trans,lower_limit_Bonus_trans,airline['Flight_trans_12'])))
sns.boxplot(airline.Flight_trans_12);plt.title('Boxplot');plt.show()

## Using Min-Max method for scaling data#
def norm_fun(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
airline_norm = norm_fun(airline.iloc[ : ,0:])

airline_norm.isnull().sum()
airline_norm.isna().sum()
#there are few NAn value in data set#
airline_norm2=airline_norm.replace(to_replace=np.nan, value=0)
airline_norm2.isna().sum()
#for linkage for Hiearchical clustering#
from scipy.cluster.hierarchy import linkage    
import scipy.cluster.hierarchy as sch

##Linkage and plotting Dendogram ##
linkage_single = linkage(airline_norm2,method='single', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_single,leaf_rotation=0, leaf_font_size=10)
plt.show()

linkage_single = linkage(airline_norm2,method='single', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_single,leaf_rotation=0, leaf_font_size=10)
plt.show()

linkage_complete = linkage(airline_norm2,method='complete', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for complete linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_complete,leaf_rotation=0, leaf_font_size=10)
plt.show()

linkage_average = linkage(airline_norm2,method='average', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for average linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_average,leaf_rotation=0, leaf_font_size=10)
plt.show()

linkage_centroid = linkage(airline_norm2,method='centroid', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for centroid linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_centroid,leaf_rotation=0, leaf_font_size=10)
plt.show()

from sklearn.cluster import AgglomerativeClustering
#lets find Hiearchical clustering using Agglomerative clusting#
sk_linkage_single =  AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage= 'single').fit(airline_norm2)
cluster_airline_single= pd.Series(sk_linkage_single.labels_)
airline['cluster']= cluster_airline_single

sk_linkage_complete =  AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage= 'complete').fit(airline_norm2)
cluster_airline_complete= pd.Series(sk_linkage_complete.labels_)
airline['cluster']= cluster_airline_complete

sk_linkage_average =  AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage= 'average').fit(airline_norm2)
cluster_airline_average= pd.Series(sk_linkage_average.labels_)
airline['cluster']= cluster_airline_average

sk_linkage_ward =  AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage= 'ward').fit(airline_norm2)
cluster_airline_ward= pd.Series(sk_linkage_ward.labels_)
airline['cluster']= cluster_airline_ward

##indexing the columns#
airline= airline.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]]

##aggregating into 3 clusters and visualizing#
airline.iloc[:, 0].groupby(airline.cluster).mean()
##extracting final file as CSV file#
airline.to_csv('new_airline', encoding = "utf-8")
import os
os.getcwd()

#####################################problem 2########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering

#Let load Data frame ##
crime_data = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Dataset_Assignment Clustering/crime_data.csv')
crime_data.columns

#cheking null and NA value 
crime_data.isna().sum()
crime_data.isnull().sum()

#Duplicate data checking#
dups1 = crime_data.duplicated()
sum(dups1)

#plotting Boxplot for checking outliers#
crime_data.columns
sns.boxplot(crime_data.Murder);plt.title('Boxplot');plt.show()
sns.boxplot(crime_data.Assault);plt.title('Boxplot');plt.show()
sns.boxplot(crime_data.UrbanPop);plt.title('Boxplot');plt.show()
sns.boxplot(crime_data.Rape);plt.title('Boxplot');plt.show()
##Scatter plot using Bivariate##
plt.scatter(crime_data['Murder'], crime_data['Assault'])
plt.scatter(crime_data['UrbanPop'], crime_data['Rape'])

#there is an outlier in columns named Rape#
#Removing outliers#
IQR = crime_data['Rape'].quantile(0.75)-crime_data['Rape'].quantile(0.25)
lower_limit_Rape = crime_data['Rape'].quantile(0.25)-(IQR*1.5)
upper_limit_Rape = crime_data['Rape'].quantile(0.75)+(IQR*1.5)
crime_data['Rape']=pd.DataFrame(np.where(crime_data['Rape']>upper_limit_Rape,upper_limit_Rape,
                                         np.where(crime_data['Rape']<lower_limit_Rape,lower_limit_Rape,crime_data['Rape'])))
sns.boxplot(crime_data.Rape);plt.title('Boxplot');plt.show()

#defining scaling function usinf Min max method#
def norm_fun(i):
    x= (i-i.min())/(i.max()-i.min())
    return(x)
#normalizing the data
crime_data_norm = norm_fun(crime_data.iloc[: , 1:])

## Hierachical clustering using 4 types of linkage##
linkage_single = linkage(crime_data_norm,method='single', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_single,leaf_rotation=0, leaf_font_size=10)
plt.show()

linkage_complete = linkage(crime_data_norm,method='complete', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for complete linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_complete,leaf_rotation=0, leaf_font_size=10)
plt.show()

linkage_average = linkage(crime_data_norm,method='average', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for average linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_average,leaf_rotation=0, leaf_font_size=10)
plt.show()

linkage_centroid = linkage(crime_data_norm,method='centroid', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for centroid linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_centroid,leaf_rotation=0, leaf_font_size=10)
plt.show()

##clustering using Agglomerative clustering##
crime_single = AgglomerativeClustering(n_clusters=3 , linkage="single" , affinity="euclidean").fit(crime_data_norm)
cluster_crime_single = pd.Series(crime_single.labels_)
crime_data["cluster"] = cluster_crime_single

crime_complete = AgglomerativeClustering(n_clusters=3 , linkage="complete" , affinity="euclidean").fit(crime_data_norm)
cluster_crime_complete= pd.Series(crime_complete.labels_)
crime_data["cluster"] = cluster_crime_complete

crime_average = AgglomerativeClustering(n_clusters=3 , linkage="average" , affinity="euclidean").fit(crime_data_norm)
cluster_crime_average = pd.Series(crime_average.labels_)
crime_data["cluster"] = cluster_crime_average

crime_ward = AgglomerativeClustering(n_clusters=3 , linkage="ward" , affinity="euclidean").fit(crime_data_norm)
cluster_crime_ward = pd.Series(crime_ward.labels_)
crime_data["cluster"] = cluster_crime_ward

crime_data=crime_data.iloc[:,[5,0,1,2,3,4]]
crime_data.iloc[:,1:].groupby(crime_data.cluster).mean()

import os
##exporting final file as csv#
crime_data.to_csv('finale_crime_data', encoding = 'utf-8')
os.getcwd()

###########################################problem 3################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the data set#
telco = pd.read_excel('D:/DATA SCIENCE ASSIGNMENT/Dataset_Assignment Clustering/Telco_customer_churn.xlsx')

telco.info()
telco.describe

## lets drop count and category ##
telco.drop(['Count','Quarter'], axis = 1 , inplace = True)

## Converting categorical data by using dummy value##
telco2 = pd.get_dummies(telco)

##checking for any duplicate data #
dup1 = telco.duplicated()
sum(dup1)
telco = telco.drop_duplicates()

#importing one hot encoder package #
from sklearn.preprocessing import OneHotEncoder

OH_enc = OneHotEncoder()
telco3 = pd.DataFrame(OH_enc.fit_transform(telco).toarray())

telco.info()
from sklearn.preprocessing import LabelEncoder
#lable encoding categorical or nominal data##
L_enc =LabelEncoder()

telco['Reffered a Friend'] = pd.DataFrame(L_enc.fit_transform(telco['Referred a Friend']))
telco['Offer'] = pd.DataFrame(L_enc.fit_transform(telco['Offer']))
telco['Phone Service'] = pd.DataFrame(L_enc.fit_transform(telco['Phone Service']))
telco['Multiple Lines'] = pd.DataFrame(L_enc.fit_transform(telco['Multiple Lines']))
telco['Internet Service'] = pd.DataFrame(L_enc.fit_transform(telco['Internet Service']))
telco['Internet Type'] = pd.DataFrame(L_enc.fit_transform(telco['Internet Type']))
telco['Online Security'] = pd.DataFrame(L_enc.fit_transform(telco['Online Security']))
telco['Online Backup'] = pd.DataFrame(L_enc.fit_transform(telco['Online Backup']))
telco['Device Protection Plan'] = pd.DataFrame(L_enc.fit_transform(telco['Device Protection Plan']))
telco['Premium Tech Support'] = pd.DataFrame(L_enc.fit_transform(telco['Premium Tech Support']))
telco['Streaming TV'] = pd.DataFrame(L_enc.fit_transform(telco['Streaming TV']))
telco['Streaming Movies'] = pd.DataFrame(L_enc.fit_transform(telco['Streaming Movies']))
telco['Streaming Music'] = pd.DataFrame(L_enc.fit_transform(telco['Streaming Music']))
telco['Unlimited Data'] = pd.DataFrame(L_enc.fit_transform(telco['Unlimited Data']))
telco['Contract'] = pd.DataFrame(L_enc.fit_transform(telco['Contract']))
telco['Paperless Billing'] = pd.DataFrame(L_enc.fit_transform(telco['Paperless Billing']))
telco['Payment Method'] = pd.DataFrame(L_enc.fit_transform(telco['Payment Method']))

##Checking for any NA or null value##
telco.isna().sum() ## there is no such values##
telco.columns

##Boxplotting and outlier analysis#univariate EDA#
sns.boxplot(telco['Tenure in Months']);plt.title('Boxplot');plt.show()
sns.boxplot(telco['Avg Monthly Long Distance Charges']);plt.title('Boxplot');plt.show()
sns.boxplot(telco['Avg Monthly GB Download']);plt.title('Boxplot');plt.show()
sns.boxplot(telco['Monthly Charge']);plt.title('Boxplot');plt.show()
sns.boxplot(telco['Total Charges']);plt.title('Boxplot');plt.show()
sns.boxplot(telco['Total Refunds']);plt.title('Boxplot');plt.show()
sns.boxplot(telco['Total Extra Data Charges']);plt.title('Boxplot');plt.show()
sns.boxplot(telco['Total Long Distance Charges']);plt.title('Boxplot');plt.show()
sns.boxplot(telco['Total Revenue']);plt.title('Boxplot');plt.show()

##Scatter plot## Bivariate EDA##
plt.scatter(telco['Tenure in Months'], telco['Total Extra Data Charges'])
plt.scatter(telco['Monthly Charge'], telco['Avg Monthly Long Distance Charges'])
plt.scatter(telco['Total Long Distance Charges'], telco['Total Revenue'])

##lets remove outlier ##

IQR = telco['Avg Monthly GB Download'].quantile(0.75)-telco['Avg Monthly GB Download'].quantile(0.25)
lower_limit_Avg_Monthly_GB_Download = telco['Avg Monthly GB Download'].quantile(0.25)- (IQR*1.5)
upper_limit_Avg_Monthly_GB_Download = telco['Avg Monthly GB Download'].quantile(0.75)+ (IQR*1.5)
telco['Avg Monthly GB Download'] =pd.DataFrame(np.where(telco['Avg Monthly GB Download']>upper_limit_Avg_Monthly_GB_Download,upper_limit_Avg_Monthly_GB_Download,
                                                    np.where(telco['Avg Monthly GB Download']< lower_limit_Avg_Monthly_GB_Download,lower_limit_Avg_Monthly_GB_Download,telco['Avg Monthly GB Download'])))
sns.boxplot(telco['Avg Monthly GB Download']);plt.title('Boxplot');plt.show()

IQR = telco['Total Refunds'].quantile(0.75)-telco['Total Refunds'].quantile(0.25)
lower_limit_Total_Refunds = telco['Total Refunds'].quantile(0.25)- (IQR*1.5)
upper_limit_Total_Refunds = telco['Total Refunds'].quantile(0.75)+ (IQR*1.5)
telco['Total Refunds'] =pd.DataFrame(np.where(telco['Total Refunds']>upper_limit_Total_Refunds,upper_limit_Total_Refunds,
                                                    np.where(telco['Total Refunds']< lower_limit_Total_Refunds,lower_limit_Total_Refunds, telco['Total Refunds'])))
sns.boxplot(telco['Total Refunds']);plt.title('Boxplot');plt.show()

IQR = telco['Total Extra Data Charges'].quantile(0.75)-telco['Total Extra Data Charges'].quantile(0.25)
lower_limit_Total_Extra_Data_Charges = telco['Total Extra Data Charges'].quantile(0.25)- (IQR*1.5)
upper_limit_Total_Extra_Data_Charges = telco['Total Extra Data Charges'].quantile(0.75)+ (IQR*1.5)
telco['Total Extra Data Charges'] =pd.DataFrame(np.where(telco['Total Extra Data Charges']>upper_limit_Total_Extra_Data_Charges,upper_limit_Total_Extra_Data_Charges,
                                                    np.where(telco['Total Extra Data Charges']< lower_limit_Total_Extra_Data_Charges,lower_limit_Total_Extra_Data_Charges, telco['Total Extra Data Charges'])))
sns.boxplot(telco['Total Extra Data Charges']);plt.title('Boxplot');plt.show()

IQR = telco['Total Long Distance Charges'].quantile(0.75)-telco['Total Long Distance Charges'].quantile(0.25)
lower_limit_Total_Long_Distance_Charges = telco['Total Long Distance Charges'].quantile(0.25)- (IQR*1.5)
upper_limit_Total_Long_Distance_Charges = telco['Total Long Distance Charges'].quantile(0.75)+ (IQR*1.5)
telco['Total Long Distance Charges'] =pd.DataFrame(np.where(telco['Total Long Distance Charges']>upper_limit_Total_Long_Distance_Charges,upper_limit_Total_Long_Distance_Charges,
                                                    np.where(telco['Total Long Distance Charges']< lower_limit_Total_Long_Distance_Charges,lower_limit_Total_Long_Distance_Charges , telco['Total Long Distance Charges'])))
sns.boxplot(telco['Total Long Distance Charges']);plt.title('Boxplot');plt.show()

IQR = telco['Total Revenue'].quantile(0.75)-telco['Total Revenue'].quantile(0.25)
lower_limit_Total_Revenue = telco['Total Revenue'].quantile(0.25)- (IQR*1.5)
upper_limit_Total_Revenue = telco['Total Revenue'].quantile(0.75)+ (IQR*1.5)
telco['Total Revenue'] =pd.DataFrame(np.where(telco['Total Revenue']>upper_limit_Total_Revenue,upper_limit_Total_Revenue,
                                                    np.where(telco['Total Revenue']< lower_limit_Total_Revenue,lower_limit_Total_Revenue , telco['Total Revenue'])))
sns.boxplot(telco['Total Revenue']);plt.title('Boxplot');plt.show()

## All outlier has been removed ##

#define scaling funct standarization##
def std_fun(i):
    x=(i-i.min())/(i.std())
    return(x)

#standardization##
telco_norm = std_fun(telco2)
str(telco_norm)

#checking for NA or null value again ##
telco2.isna().sum()
telco2.isnull().sum()

##importing packages for H clustering#
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage

##hiearchical using different linkages##
telco_single_linkage = linkage(telco_norm, method='single', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hiearchical clustering using single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_single_linkage,leaf_font_size=10,leaf_rotation=0)
plt.show()


telco_complete_linkage = linkage(telco_norm, method='complete', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hiearchical clustering using complete linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_complete_linkage,leaf_font_size=10,leaf_rotation=0)
plt.show()

telco_average_linkage = linkage(telco_norm, method='average', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hiearchical clustering using average linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_average_linkage,leaf_font_size=10,leaf_rotation=0)
plt.show()

telco_centroid_linkage = linkage(telco_norm, method='centroid', metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hiearchical clustering using centroid linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_centroid_linkage,leaf_font_size=10,leaf_rotation=0)
plt.show()


from sklearn.cluster import AgglomerativeClustering
##Clustering using Agglomerative clustering##

telco_single = AgglomerativeClustering(n_clusters=3,linkage='single', affinity='euclidean').fit(telco_norm)
cluster_telco_single = pd.Series(telco_single.labels_)
telco['cluster']= cluster_telco_single

telco_complete = AgglomerativeClustering(n_clusters=3,linkage='complete', affinity='euclidean').fit(telco_norm)
cluster_telco_complete = pd.Series(telco_complete.labels_)
telco['cluster']= cluster_telco_complete

telco_average = AgglomerativeClustering(n_clusters=3,linkage='average', affinity='euclidean').fit(telco_norm)
cluster_telco_average = pd.Series(telco_average.labels_)
telco['cluster']= cluster_telco_average

telco_ward = AgglomerativeClustering(n_clusters=3,linkage='ward', affinity='euclidean').fit(telco_norm)
cluster_telco_ward = pd.Series(telco_ward.labels_)
telco['cluster']= cluster_telco_ward

telco.iloc[:, 0:29].groupby(telco.cluster).mean()

import os
telco.to_csv('final_telco.csv', encoding='utf-8')
os.getcwd()

########################## Problem 4#########################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##loading dataframe ##

auto = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Dataset_Assignment Clustering/AutoInsurance.csv')

auto.info()
auto.describe
#droping customer column as it has minimal insights#
auto.drop(['Customer'], axis=1 , inplace = True)

auto_new = auto.iloc[:, 1:]

auto_new.isna().sum()
auto_new.isnull().sum()

auto_new.columns

##check duplicate value#

dupl = auto_new.duplicated()
sum(dupl) ## total 512
#removing duplicate##
auto_new = auto_new.drop_duplicates()

##boxplotting and outlier analysis univariates EDA##
sns.boxplot(auto_new['Customer Lifetime Value']);plt.title('Boxplot');plt.show()
sns.boxplot(auto_new['Income']);plt.title('Boxplot');plt.show()
sns.boxplot(auto_new['Monthly Premium Auto']);plt.title('Boxplot');plt.show()
sns.boxplot(auto_new['Months Since Last Claim']);plt.title('Boxplot');plt.show()
sns.boxplot(auto_new['Months Since Policy Inception']);plt.title('Boxplot');plt.show()
sns.boxplot(auto_new['Total Claim Amount']);plt.title('Boxplot');plt.show()
sns.boxplot(auto_new['Number of Policies']);plt.title('Boxplot');plt.show()

##scatterplot bivariates##

plt.scatter(auto_new['Customer Lifetime Value'], auto_new['Income'])
plt.scatter(auto_new['Monthly Premium Auto'], auto_new['Months Since Last Claim'])
plt.scatter(auto_new['Months Since Policy Inception'], auto_new['Total Claim Amount'])


##Outlier removal##

IQR = auto_new['Customer Lifetime Value'].quantile(0.75)-auto_new['Customer Lifetime Value'].quantile(0.25)
lower_limit_Customer_Lifetime_Value = auto_new['Customer Lifetime Value'].quantile(0.25)- (IQR*1.5)
upper_limit_Customer_Lifetime_Value = auto_new['Customer Lifetime Value'].quantile(0.75)+ (IQR*1.5)
auto_new['Customer Lifetime Value'] =pd.DataFrame(np.where(auto_new['Customer Lifetime Value']>upper_limit_Customer_Lifetime_Value,upper_limit_Customer_Lifetime_Value,
                                                    np.where(auto_new['Customer Lifetime Value']< lower_limit_Customer_Lifetime_Value,lower_limit_Customer_Lifetime_Value,auto_new['Customer Lifetime Value'])))
sns.boxplot(auto_new['Customer Lifetime Value']);plt.title('Boxplot');plt.show()

IQR = auto_new['Monthly Premium Auto'].quantile(0.75)-auto_new['Monthly Premium Auto'].quantile(0.25)
lower_limit_Monthly_Premium_Auto = auto_new['Monthly Premium Auto'].quantile(0.25)- (IQR*1.5)
upper_limit_Monthly_Premium_Auto = auto_new['Monthly Premium Auto'].quantile(0.75)+ (IQR*1.5)
auto_new['Monthly Premium Auto'] =pd.DataFrame(np.where(auto_new['Monthly Premium Auto']>upper_limit_Monthly_Premium_Auto,upper_limit_Monthly_Premium_Auto,
                                                    np.where(auto_new['Monthly Premium Auto']< lower_limit_Monthly_Premium_Auto,lower_limit_Monthly_Premium_Auto,auto_new['Monthly Premium Auto'])))
sns.boxplot(auto_new['Monthly Premium Auto']);plt.title('Boxplot');plt.show()

IQR = auto_new['Total Claim Amount'].quantile(0.75)-auto_new['Total Claim Amount'].quantile(0.25)
lower_limit_Total_Claim_Amount = auto_new['Total Claim Amount'].quantile(0.25)- (IQR*1.5)
upper_limit_Total_Claim_Amount = auto_new['Total Claim Amount'].quantile(0.75)+ (IQR*1.5)
auto_new['Total Claim Amount'] =pd.DataFrame(np.where(auto_new['Total Claim Amount']>upper_limit_Total_Claim_Amount,upper_limit_Total_Claim_Amount,
                                                    np.where(auto_new['Total Claim Amount']< lower_limit_Total_Claim_Amount,lower_limit_Total_Claim_Amount,auto_new['Total Claim Amount'])))
sns.boxplot(auto_new['Total Claim Amount']);plt.title('Boxplot');plt.show()


##Get dummy variables##

dummy_auto= pd.get_dummies(auto_new)

##define scaling ##

def norm_fun(i):
    x=(i-i.min()/(i.max()-i.min()))
    return (x) 

#normalizing data#
auto_norm = norm_fun(dummy_auto)

##checking Nan or null value again
auto_norm.isnull().sum()
auto_norm.isna().sum()

#removing null or na value##

auto_norm1 =auto_norm.replace(to_replace=np.nan, value=0)
auto_norm1.isna().sum()
auto_norm1.isnull().sum()
from sklearn.cluster import AgglomerativeClustering

#Clustering using Agglomerative##

auto_single = AgglomerativeClustering(n_clusters=3 ,linkage= 'single',affinity='euclidean').fit(auto_norm1)
cluster_auto_single= pd.Series(auto_single.labels_)
auto_new['cluster'] = cluster_auto_single

auto_complete = AgglomerativeClustering(n_clusters=3 ,linkage= 'complete',affinity='euclidean').fit(auto_norm1)
cluster_auto_complete= pd.Series(auto_complete.labels_)
auto_new['cluster'] = cluster_auto_complete

auto_average = AgglomerativeClustering(n_clusters=3 ,linkage= 'average',affinity='euclidean').fit(auto_norm1)
cluster_auto_average= pd.Series(auto_average.labels_)
auto_new['cluster'] = cluster_auto_average

auto_ward = AgglomerativeClustering(n_clusters=3 ,linkage= 'ward',affinity='euclidean').fit(auto_norm1)
cluster_auto_ward= pd.Series(auto_ward.labels_)
auto_new['cluster'] = cluster_auto_ward

auto_new.iloc[: ,:23].groupby(auto_new.cluster).mean()
import os

auto_new.to_csv("final_auto.csv" , encoding = 'utf-8')
os.getcwd()


