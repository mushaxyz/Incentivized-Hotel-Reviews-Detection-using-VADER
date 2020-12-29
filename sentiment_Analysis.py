# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:02:56 2020

@author: Ahmmad Musha
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

# Importing the dataset
dataset1 = pd.read_csv('dataset_1.csv')
dataset2 = pd.read_csv('dataset_2.csv')
dataset3 = pd.read_csv('dataset_3.csv')

df1 = dataset1.iloc[1:, [6,2,8,14]] 
df2 = dataset2.iloc[1:, [11,6,13,18]]
df3 = dataset3.iloc[1:, [11,6,13,19]]

dataset = pd.concat([df1,df2,df3])
hotelName = dataset.iloc[:, 0].values.tolist()
temp = dataset.values
#y = dataset.iloc[1:, 14].values.tolist()


hotelList = []
newHotelList = []
finalDataset = []

# Creating Specific Hotel List
def hotelName_to_hotelList(hotelName, hotelList):
    for hotel in hotelName:
        if hotel not in hotelList:
            hotelList.append(hotel)

hotelName_to_hotelList(hotelName, hotelList)


# Creating Hotel List which reviews more than 100
for i in hotelList:
   if hotelName.count(i) > 100 :
       newHotelList.append(i)


# Short new Hotel list
newHotelList = sorted(newHotelList)


# Creating the Final DataSet which reviews more than 100 
for j in range(len(temp)):
    if temp[j][0] in newHotelList:
        finalDataset.append(temp[j])
    
    

# Creating a Final Data Frame after Data PreProcessing 
finalDataframe = pd.DataFrame(finalDataset, columns =['Name', 'City', 'Country Code', 'Reviews'])


# Function of SentimentIntensityAnalyzer
def sentiment_scores(sentence): 
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 

	# polarity_scores method of SentimentIntensityAnalyzer 
	# oject gives a sentiment dictionary. 
	# which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence)
    
    print("Review:", sentence)
    print("Overall sentiment dictionary is : ", sentiment_dict) 
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 

    # decide sentiment as positive, negative and neutral 
    # result = 'neu'
    if sentiment_dict['compound'] >= 0.05 : 
        result = 'pos'
        print("Positive") 

    elif sentiment_dict['compound'] <= - 0.05 : 
        result = 'neg'
        print("Negative") 
    else : 
        result = 'neu'
        print("Neutral")
    
    sentiment_dict['result'] = result
    
    return sentiment_dict
        



# Working on final DataSet
# Creating DataFrame of Result
for k in range(len(finalDataset)): 
    a = sentiment_scores(finalDataset[k][3])  # sentence pass in sentiment_scores function
    if k == 0:
        resultDataFrame = pd.DataFrame(a, index= [0])
    else:
        c = pd.DataFrame(a, index= [k])
        resultDataFrame = pd.concat([resultDataFrame, c], axis = 0)
    
    

# Merge finalDataFrame and resultDataFrame
merge_data = pd.concat([finalDataframe, resultDataFrame], axis = 1)
# Sort Data 
data = merge_data.sort_values('Name')
data = data.reset_index()
data = data.iloc[:, 1:]
obj_data = data.values   # Creating object



# Distribute individual hotel in a dictionary
dic_all_hotel = {}
for i in range(len(newHotelList)):
    te = data[data['Name'] == newHotelList[i]] #self.getGroup(selected, header+i)
    te = te.reset_index()
    dic_all_hotel["Hotel_" + str(i)] = te.iloc[:, 1:]




# Mean of compound 
for i in range(len(newHotelList)):
    dic_all_hotel["Hotel_" + str(i)]['Mean of compound'] = np.mean(dic_all_hotel["Hotel_" + str(i)]['compound'])

# Median of compound 
for i in range(len(newHotelList)):
    dic_all_hotel["Hotel_" + str(i)]['Median of compound'] = np.median(dic_all_hotel["Hotel_" + str(i)]['compound'])

# Standard daviation with respect to Mean of compound
for i in range(len(newHotelList)):
    dic_all_hotel["Hotel_" + str(i)]['Standard Deviation w r to mean'] = np.std(dic_all_hotel["Hotel_" + str(i)]['compound'])


# Standard daviation with respect to Median of compound
for i in range(len(newHotelList)):
    sum_of_difference = 0
    for j in range(dic_all_hotel["Hotel_" + str(i)].shape[0]):
        sum_of_difference += np.square(dic_all_hotel["Hotel_" + str(i)]['compound'][j] - dic_all_hotel["Hotel_" + str(i)]['Median of compound'][0])
        dic_all_hotel["Hotel_" + str(i)]['Standard Deviation w r to median'] = np.sqrt(np.abs(sum_of_difference/dic_all_hotel["Hotel_" + str(i)].shape[0]))
        


for i in range(len(newHotelList)):
    dic_all_hotel["Hotel_" + str(i)]['Difference Compound and mean or median from smallest Std Dev'] = 0
    for j in range(dic_all_hotel["Hotel_" + str(i)].shape[0]):
        # find minimum std dev from mean and median
        if dic_all_hotel["Hotel_" + str(i)]['Standard Deviation w r to median'][0] > dic_all_hotel["Hotel_" + str(i)]['Standard Deviation w r to mean'][0]:
#            small_std_value_bt_mean_median = dic_all_hotel["Hotel_" + str(i)]['Standard Deviation w r to mean'][0]
            mean_or_median_from_small_std  = dic_all_hotel["Hotel_" + str(i)]['Mean of compound'][0]
        elif dic_all_hotel["Hotel_" + str(i)]['Standard Deviation w r to median'][0] < dic_all_hotel["Hotel_" + str(i)]['Standard Deviation w r to mean'][0]:
#            small_std_value_bt_mean_median = dic_all_hotel["Hotel_" + str(i)]['Standard Deviation w r to median'][0]
            mean_or_median_from_small_std  = dic_all_hotel["Hotel_" + str(i)]['Median of compound'][0]
        elif dic_all_hotel["Hotel_" + str(i)]['Standard Deviation w r to median'][0] == dic_all_hotel["Hotel_" + str(i)]['Standard Deviation w r to mean'][0]:
#            small_std_value_bt_mean_median = dic_all_hotel["Hotel_" + str(i)]['Standard Deviation w r to mean'][0]
            mean_or_median_from_small_std  = dic_all_hotel["Hotel_" + str(i)]['Mean of compound'][0]
        dic_all_hotel["Hotel_" + str(i)].loc[[j], ['Difference Compound and mean or median from smallest Std Dev']] = (np.abs(dic_all_hotel["Hotel_" + str(i)]['compound'][j] - mean_or_median_from_small_std))  



        
 
# Incentivized > 0.5   
for i in range(len(newHotelList)): 
    dic_all_hotel["Hotel_" + str(i)]['Output'] = 0
    dic_all_hotel["Hotel_" + str(i)]['Class_value_of_Output_Label'] =  0
    for j in range(dic_all_hotel["Hotel_" + str(i)].shape[0]):
        if dic_all_hotel["Hotel_" + str(i)]['Difference Compound and mean or median from smallest Std Dev'][j] <= 0.5:
            dic_all_hotel["Hotel_" + str(i)].loc[[j], ['Output']] = "Proper Comment"
            dic_all_hotel["Hotel_" + str(i)].loc[[j], ['Class_value_of_Output_Label']] = 0
        else:
            dic_all_hotel["Hotel_" + str(i)].loc[[j], ['Output']] = "Incentivized"
            dic_all_hotel["Hotel_" + str(i)].loc[[j], ['Class_value_of_Output_Label']] = 1
        

# Convert Dic to DataFrame
for i in range(len(newHotelList)):
    if i == 0:
        final_result_dataframe = pd.DataFrame.from_dict(dic_all_hotel["Hotel_" + str(0)])
    else:
        e = pd.DataFrame.from_dict(dic_all_hotel["Hotel_" + str(i)])
        final_result_dataframe = pd.concat([final_result_dataframe, e], ignore_index = True)



# Applying Machine Learning Approach

# DataSet
X = final_result_dataframe.iloc[:, [4,5,6,7,13]]
y = final_result_dataframe.iloc[:, 15].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)        


# import the Machine Learning Algorithm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



# Create different classifiers.
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators = 2, criterion = 'entropy', random_state = 0),
    'KNN': KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
    'SVM': SVC(kernel = 'linear', random_state = 0)
}


n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, y_train)   # Fitting the Machine learning Algorithm to the Training set

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy for %s: %0.1f%% " % (name, accuracy * 100))
    print("The confusion Matrix for " +name+": \n", cm)



##### Graphical Visualise  #####

# Create DataFrame for graph
for i in range(len(newHotelList)):
    graph_dic_all_hotel = {}
    count_incentivized_comment = count_proper_comment = 0
    
        # count pos,neg, neu
    pos_count = neg_count = neu_count = 0
    
    
    if dic_all_hotel["Hotel_" + str(i)]["Name"][0] == newHotelList[i]:
        for j in range(dic_all_hotel["Hotel_" + str(i)].shape[0]):
            if dic_all_hotel["Hotel_" + str(i)]['Output'][j] == "Incentivized":
                count_incentivized_comment += 1
            else:
                count_proper_comment += 1
                
            # count pos,neg, neu
            if dic_all_hotel["Hotel_" + str(i)]['result'][j] == "pos":
                pos_count += 1
            elif dic_all_hotel["Hotel_" + str(i)]['result'][j] == "neg":
                neg_count += 1
            else:
                neu_count += 1   
                
        # data for show_stat(stat) function
        graph_dic_all_hotel['hotel'] = dic_all_hotel["Hotel_" + str(i)]["Name"][0]
        graph_dic_all_hotel["total_comment"] = count_proper_comment + count_incentivized_comment
        graph_dic_all_hotel["proper_comment"] = count_proper_comment
        graph_dic_all_hotel["incentivized_comment"] = count_incentivized_comment
        graph_dic_all_hotel["pos_count"] = pos_count
        graph_dic_all_hotel["neg_count"] = neg_count
        graph_dic_all_hotel["neu_count"] = neu_count                
        graph_dic_all_hotel['Country Code'] = dic_all_hotel["Hotel_" + str(i)]["Country Code"][0]
        graph_dic_all_hotel['City'] = dic_all_hotel["Hotel_" + str(i)]["City"][0]
        
        
    if i == 0:
        graph_dataframe_all_hotel = pd.DataFrame(graph_dic_all_hotel, index=[i])  
    else:
        f = pd.DataFrame(graph_dic_all_hotel, index=[i]) 
        graph_dataframe_all_hotel = pd.concat([graph_dataframe_all_hotel, f], ignore_index = True)



#
##Visualization Geographical plot
#import plotly.plotly as py
#import plotly.graph_objs as go 
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#
#init_notebook_mode(connected=True)
#data_for_graph = dict(type='choropleth',          
#                      colorscale = 'YlOrRd', 
#                      locations = graph_dataframe_all_hotel['Country Code'],
#                      z = graph_dataframe_all_hotel['incentivized_comment'],
#                      locationmode = 'USA-states',
#                      text = graph_dataframe_all_hotel['City'],
#                      marker = dict(line = dict(color = 'rgb(12,12,12)',width = 1)),
#                      colorbar = {'title':"Incentivized Comment"}
#                      ) 
#
#layout = dict(title = 'Incentivized Comment in US',
#              geo = dict(scope='usa',
#                         showlakes = True,
#                         lakecolor = 'rgb(85,173,240)')
#             )
#choromap = go.Figure(data = [data_for_graph],layout = layout)
## iplot(choromap)
#plot(choromap, auto_open=True)
#

##################################################################



# all_hotel_summary_barchart 
total_review_all_hotel = np.sum(graph_dataframe_all_hotel['total_comment'])
real_review_all_hotel = np.sum(graph_dataframe_all_hotel['proper_comment'])
insentivized_review_all_hotel = np.sum(graph_dataframe_all_hotel['incentivized_comment'])

labels = ["Total Review", "Real Review", "Insectivized Review"]
values = [total_review_all_hotel, real_review_all_hotel, insentivized_review_all_hotel]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(labels,values)
plt.title('Bar Chart for all hotel Summary') 
plt.xticks(labels, rotation='vertical')
plt.show()


# pos_neg_neu_piechart
pos_review = np.sum(graph_dataframe_all_hotel['pos_count'])
neg_review = np.sum(graph_dataframe_all_hotel['neg_count'])
neu_review = np.sum(graph_dataframe_all_hotel['neu_count'])

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
labels = 'Positive', 'Negative', 'Neutral'
sizes = [pos_review, neg_review, neu_review]
ax.pie(sizes, labels = labels,autopct='%1.2f%%')
plt.title('Pie Chart for Positive Negative and Neutral Review') 
plt.show()


# Bar Chart of Insentivized Review for all hotel
hotel =  graph_dataframe_all_hotel['hotel']
incentivized_comment = graph_dataframe_all_hotel['incentivized_comment']
fig = plt.figure()
ax = fig.add_axes([0,0,3,3])
ax.bar(hotel,incentivized_comment)
plt.title('Number of Incentivezed Review vs Hotel Name') 
plt.ylabel('Number of Insentivized Review')
plt.xticks(hotel, rotation='vertical')
plt.show()


# Bar Chart of Proper Review for all hotel
hotel =  graph_dataframe_all_hotel['hotel']
proper_comment = graph_dataframe_all_hotel['proper_comment']
fig = plt.figure()
ax = fig.add_axes([0,0,3,3])
ax.bar(hotel,proper_comment)
plt.title('Number of Proper Review vs Hotel Name') 
plt.ylabel('Number of Proper Review')
plt.xticks(hotel, rotation='vertical')
plt.show()



#Scatter_and_plot incentivized_vs_hotel
hotel =  graph_dataframe_all_hotel['hotel']
incentivized_comment = graph_dataframe_all_hotel['incentivized_comment']
fig = plt.figure()
ax = fig.add_axes([0,0,3,3])
ax.scatter(hotel, incentivized_comment)
ax.plot(hotel, incentivized_comment, color='green')
plt.title('Number of Incentivezed Review vs Hotel Name') 
plt.ylabel('Number of Insentivized Review')
plt.xticks(hotel, rotation='vertical')
plt.show()
 

#Scatter_and_plot real_vs_hotel
hotel =  graph_dataframe_all_hotel['hotel']
proper_comment = graph_dataframe_all_hotel['proper_comment']
fig = plt.figure()
ax = fig.add_axes([0,0,3,3])
ax.scatter(hotel, proper_comment)
ax.plot(hotel, proper_comment, color='green')
plt.title('Number of Proper Review vs Hotel Name') 
plt.ylabel('Number of Proper Review')
plt.xticks(hotel, rotation='vertical')
plt.show()


# real_vs_hotel_histogram
x = np.arange(len(hotel))
fig = plt.figure()
ax = fig.add_axes([0,0,3,3])
width = 1.0
ax.bar(x, proper_comment, width, label='Proper Review')
ax.set_ylabel('Number of Proper Review')
ax.set_title('Histogram for Proper Review vs Hotel Name')
ax.set_xticks(x)
ax.set_xticklabels(hotel, rotation=90)
ax.legend()



# Incentivized_vs_hotel_histogram
x = np.arange(len(hotel))
fig = plt.figure()
ax = fig.add_axes([0,0,3,3])
width = 1.0
ax.bar(x, incentivized_comment, width, label='Incentivized Review')
ax.set_ylabel('Number of Incentivized Review')
ax.set_title('Histogram for Incentivized Review vs Hotel Name')
ax.set_xticks(x)
ax.set_xticklabels(hotel, rotation=90)
ax.legend()



#real_insetivized_vs_hotel_barchart
fig = plt.figure()
ax = fig.add_axes([0,0,3,3])
width = 0.35
ax.bar(x - width/2, proper_comment, width, label='Proper Review')
ax.bar(x + width/2, incentivized_comment, width, label='Insentivized Review')
ax.set_ylabel('Number of Proper and Incentivized Review')
ax.set_title('Number of Proper and Incentivized Review vs Hotel Name')
ax.set_xticks(x)
ax.set_xticklabels(hotel, rotation=90)
ax.legend()



# total_real_vs_hotel_variance
tot_comment = graph_dataframe_all_hotel['total_comment']
fig = plt.figure()
ax = fig.add_axes([0,0,3,3])

ax.scatter(hotel, tot_comment)
ax.plot(hotel, tot_comment, color='green', label='Total Review')

ax.scatter(hotel, proper_comment)
ax.plot(hotel, proper_comment, color='red', label='Proper Review')

ax.set_ylabel('Number of Total and Proper Review')
ax.set_title('Number of Proper and Total Review vs Hotel Name')
ax.set_xticklabels(hotel, rotation=90)
ax.legend()



# total_incentivised_vs_hotel_variance
fig = plt.figure()
ax = fig.add_axes([0,0,3,3])

ax.scatter(hotel, tot_comment)
ax.plot(hotel, tot_comment, color='green', label='Total Review')

ax.scatter(hotel, incentivized_comment)
ax.plot(hotel, incentivized_comment, color='red', label='Incentivised Review')

ax.set_ylabel('Number of Total and Incentivised Review')
ax.set_title('Number of Incentivised and Total Review vs Hotel Name')
ax.set_xticklabels(hotel, rotation=90)
ax.legend()



# pos_neg_neu_vs_hotel_barchart_separated
pos = graph_dataframe_all_hotel['pos_count']
neg = graph_dataframe_all_hotel['neg_count']
neu = graph_dataframe_all_hotel['neu_count']
fig = plt.figure()
ax = fig.add_axes([0,0,3,3])
width = 0.35
ax.bar(x - width/3, pos, width, label='Positive Review')
ax.bar(x + width/3, neg, width, label='Negative Review')
ax.bar(x + width/3, neu, width, label='Neutral Review')
ax.set_ylabel('Number of Positive Negative and Neutral Review')
ax.set_title('Number of Positive Negative and Neutral Review vs Hotel Name')
ax.set_xticks(x)
ax.set_xticklabels(hotel, rotation=90)
ax.legend()




# For CSV File
FINAL_FILE_NAME = 'OutputResult.csv'
final_result_dataframe.to_csv(FINAL_FILE_NAME)

# Writing Excel sheet
writer = pd.ExcelWriter('Output Result.xlsx')
final_result_dataframe.to_excel(writer,'Sheet1',index=True)
writer.save()

###---------------------------------------- The End------------------------------###