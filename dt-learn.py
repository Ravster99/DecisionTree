
# coding: utf-8

# In[15]:


#Importing important libraries.
#Import sys and scipy.io for Linux.
import sys
from scipy.io import arff
#import arff for Windows.
#import arff
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[16]:


#Some global variable declaration.
global Set_of_nodes
Set_of_nodes = []
del Set_of_nodes[:]
accuracy_count = 0


# In[ ]:


########Training data set.############################
########Lodaing of data in Linux.#####################
raw_data = arff.loadarff(open(sys.argv[1])) # Loading the training dataset
training_data = pd.DataFrame(raw_data[0])  # Saving as a Pandas Dataframe
training_data = training_data.apply(pd.to_numeric, errors='ignore') # Converting the relavent columns to numeric
# Creating a list of the features
feature_list = [] 
for x in raw_data['attributes']:
    feature_list.append(x[0])


# In[2]:


########Testing data set.###########################
########Lodaing of data in Linux.#####################
raw_data = arff.loadarff(open(sys.argv[2])) 
testing_data = pd.DataFrame(raw_data[0])
testing_data = testing_data.apply(pd.to_numeric, errors='ignore') # Converting the relavent columns to numeric
#Creating a list of the features
feature_list = [] 
for x in raw_data['attributes']:
    feature_list.append(x[0])


# In[17]:


m = int(sys.argv[3]) # Specify the node-instance threshold
#For Windows.
#m = 20


# In[18]:


########Lodaing of data in Windows.#####################
########Training data set (but to delete).############################
#raw_data = arff.load(open('heart_train.arff')) # Loading the training datasets
#raw_data = arff.load(open('diabetes_train.arff')) # Loading the training datasets 
# Creating a list of the features
#feature_list = [] 
#for x in raw_data['attributes']:
#    feature_list.append(x[0])
#training_data = pd.DataFrame(np.array(raw_data['data']), columns = feature_list) # Saving as a Pandas Dataframe
#training_data = training_data.apply(pd.to_numeric, errors='ignore') # Converting the releavent columns to numeric
###################Testing data set (but to delete).###################################
#raw_data = arff.load(open('heart_test.arff')) # Loading the training datasets
#raw_data = arff.load(open('diabetes_test.arff')) # Loading the training datasets
# Creating a list of the features
#feature_list = [] 
#for x in raw_data['attributes']:
#    feature_list.append(x[0])
#testing_data = pd.DataFrame(np.array(raw_data['data']), columns = feature_list) # Saving as a Pandas Dataframe
#testing_data = testing_data.apply(pd.to_numeric, errors='ignore') # Converting the releavent columns to numeric


# In[19]:


def PositiveAndNegative(data): #Function to determine the number of positive and negative instances in the data.
    positive, negative = 0, 0
    for Data_Point in data['class']:
            if (Data_Point == 'positive'):
                positive+= 1
            else:
                negative+= 1
    return positive, negative


# In[20]:


def Entropy(data): #Function to determine the entrop of the data.
    DataFeature_Frequency = {} #Creating a directory to story the frequency of positive and negative instances.
    Entropy = 0.0
    Sub_Column = data['class']
    for Data_Point in Sub_Column: #Accessing every data point in the column class to determine positive/negative instances.
        if (Data_Point in DataFeature_Frequency):
            #Accessing the value of key and increasing its value by 1 every time one more element is found.
            DataFeature_Frequency[Data_Point] += 1
        else:
            #If the key is found for the first time, we assign it value to be 1.
            DataFeature_Frequency[Data_Point] = 1
            
    for Frequency in DataFeature_Frequency.values():
        Entropy += (-Frequency/len(data))*math.log(Frequency/len(data), 2)
        
    return Entropy


# In[30]:


def Split(data, feature, sub_feature, numeric_partition, condition): #Function to split the data based on particular feature.
    sub_column = data[feature]
    sub_column_number = data.columns.get_loc(feature) 
    if (sub_column.dtype==np.float64):
        if (condition == 0): #Creating split for less than (condition) on the numeric data.
            split_data = data[data.iloc[:, sub_column_number] <= numeric_partition]
        else: #Creating split for more than (condition) on the numeric data.
            split_data = data[data.iloc[:, sub_column_number] > numeric_partition]    
    else: #Creating the split for the categorical data.
        split_data = data[data.iloc[:, sub_column_number] == sub_feature]
        
    Number_of_data_remaining = len(split_data) 
    return split_data, Number_of_data_remaining


# In[22]:


def Remainder(data, feature, numeric_partition): #Function to determine entropy of child.
    Remainder = 0.0
    DataFeature_Frequency = 0
    Frequncy=0
    sub_data= data[[feature, 'class']]
    sub_column = data[feature]
    if (sub_column.dtype==np.float64): #Determine information gain for numeric variables.
        for x in range(0, 2):
            (Split_Data, Number_of_Data_Remaining) = Split(sub_data, feature, 0, numeric_partition, x)
            Sub_Split_Data = Split_Data[[feature, 'class']]
            Frequency =Split_Data['class'].count()
            Remainder += (float(Frequency) / len(sub_column) ) * Entropy(Sub_Split_Data)
    else: #Determine information for categorical variables. 
        Unique_Features = sub_column.unique()
        DataFeature_Frequency = sub_column.value_counts().to_dict()        
        for Data_Point in Unique_Features:
            Frequency = DataFeature_Frequency[Data_Point]
            (Split_Data, Number_of_Data_Remaining) = Split(sub_data, feature, Data_Point, 0, 0)
            Sub_Split_Data = Split_Data[[feature, 'class']]
            SubFeature_Frequency = Sub_Split_Data.groupby(Sub_Split_Data['class']).count()
            Remainder += (float(Frequency) / len(sub_column) ) * Entropy(Sub_Split_Data)
            
    return Remainder


# In[23]:


def AttributeSelection(data): #Function to select the attribute that provides the best information gain.
    global BestFeature
    global Numeric_Partition_return
    Numeric_Partition_return = 0
    MaxGain = 0 
    Numeric_Partition = 0
    H = Entropy(data)
    H_Data = 1
    #Loop for determining the attribute which gives the best information gain.
    for column_name in (data.columns.values.tolist()):
        if(column_name == 'class'): #To exclude the last column (class) from being executed - creates an error.
            break
        sub_column = data[column_name]
        if (sub_column.dtype == np.float64): #Determine information gain for numeric variables.
            Unique_Features = sub_column.unique()
            for Data_Point in Unique_Features: 
                IG = Remainder(data, column_name, Data_Point)   
                if (IG < H_Data):
                    H_Data = IG
                    Numeric_Partition = Data_Point          
        else:  #Determine information gain for categorical variables.
            H_Data = Remainder(data, column_name, 0)
        InfoGain = H-H_Data
        if (InfoGain>MaxGain):
            MaxGain=InfoGain
            BestFeature = column_name
            Numeric_Partition_return = Numeric_Partition

    return MaxGain, BestFeature, Numeric_Partition_return


# In[24]:


#Function which creates the decision tree.
def tree(data, m, Number_of_data_remaining, positive, negative, IG, space, Decision = "start"):
    #Checking if we've reached the leaf based on certain coditions and returning the nature of the leaf.
    if (positive == 0 or negative == 0 or Number_of_data_remaining < m or IG ==0 or len(data.columns) == 2):
        if(positive >= negative):
            value = "positive"
        else:
            value = "negative"
        Node = [Decision, value, "leaf", 0]
        Set_of_nodes.append(Node)
        return value      
    else:
        (MaxGain_tree, BestFeature_tree, Numeric_Partition_tree) = AttributeSelection(data)
        (positive, negative) = PositiveAndNegative(data)
        space+=1
        #Printing of decision tree.
        if (data[BestFeature_tree].dtype==np.float64):        
            print ("|     "*space, BestFeature_tree, "<=", Numeric_Partition_tree)
        else:
            print ("|     "*space, BestFeature_tree)
        sub_column = data[BestFeature_tree]
        if (sub_column.dtype == np.float64): #Determine entropy for numerical variables.
             for x in range(0, 2):
                (Split_Data, Number_of_Data_Remaining) = Split(data, BestFeature_tree, 0, Numeric_Partition_tree, x)
                (positive, negative) = PositiveAndNegative(Split_Data)
                IG = Entropy(Split_Data)
                Split_Data= Split_Data.drop(BestFeature_tree, 1)
                space+=1
                if (x==0):
                    Node = [Decision, BestFeature_tree, "<=", Numeric_Partition_tree]
                else:
                    Node = [Decision, BestFeature_tree, ">", Numeric_Partition_tree]
                Set_of_nodes.append(Node)
                Output = tree(Split_Data, m, Number_of_data_remaining, positive, negative, IG, space, Node)
        else:      #Determine entropy for categorical variables.       
            Unique_Features = sub_column.unique()
            space+=1
            for Data_Point in Unique_Features:
                print("|     "*space, Data_Point)               
                (Split_Data, Number_of_data_remaining) = Split(data, BestFeature_tree, Data_Point, 0, 0)
                (positive, negative) = PositiveAndNegative(Split_Data)
                IG = Entropy(Split_Data)
                Split_Data = Split_Data.drop(BestFeature_tree, 1)
                Node = [Decision, BestFeature_tree, "=", Data_Point]
                Set_of_nodes.append(Node)
                Output = tree(Split_Data, m, Number_of_data_remaining, positive, negative, IG, space, Node)


# In[25]:


#Function to retrieve the tree created by decision tree function.
#This function (tree_retrieval) helps in predicting the data.
def tree_retrieval(data_line, parentnode = "start"):
    global row_number
    global accuracy
    for nodes in Set_of_nodes:
        if (nodes[0] == parentnode):
            if (nodes[-2] == "leaf"):
                if (nodes[1]=="positive"):
                    print(row_number, "Actual: ", data_line[-1], "Predicted: ", nodes[1])
                    if (data_line[-1] == "positive"):
                        accuracy+=1
                elif (nodes[1]=="negative"):
                    print(row_number, "Actual: ", data_line[-1], "Predicted: ", nodes[1])
                    if (data_line[-1] == "negative"):
                        accuracy+=1  
            elif (nodes[-2] == "="):
                if(data_line[nodes[1]] == nodes[-1]):
                    tree_retrieval(data_line, nodes)
            elif (nodes[-2] == "<="):
                if(float(data_line[nodes[1]]) <= float(nodes[-1])):                    
                    tree_retrieval(data_line, nodes)
            elif (nodes[-2] == ">"):
                if(float(data_line[nodes[1]])> float(nodes[-1])):
                    tree_retrieval(data_line, nodes)   


# In[26]:


#Function to predict the testing data.
def prediction(data):
    print("********************  Predictions  *********************")
    print("********************************************************")
    global row_number
    row_number=1
    global accuracy
    accuracy = 0
    Total_data = len(data)
    for i in range(0, len(data)):
        tree_retrieval(data.iloc[i, :])
        row_number+=1
    accuracy_fraction = accuracy/len(data)
    print("The dataset has ", len(data),"instances. The model predicted ", accuracy," correctly.")
    print("The accuracy is: ", accuracy_fraction) 


# In[27]:


#Q2. For this part, you will plot learning curves that characterize the predictive accuracy of your learned trees
#as a function of the training set size. You will do this in two problem domains.
#The first data set involves predicting the presence or absence of heart disease.
#For this problem domain, you should use heart_train.arff as your training set and heart_test.arff as your test set.
#The second data set involves predicting whether a patient has diabetes or not.
#For this problem domain, you should use diabetes_train.arff as your training set and diabetes_test.arff as your test set.
#You should plot points for training set sizes that represent 5%, 10%, 20%, 50% and 100% of the instances in each given
#training file. For each training-set size (except the largest one), randomly draw 10 different
#training sets and evaluate each resulting decision tree model on the test set.
#For each training set size, plot the average test-set accuracy and the minimum and maximum test-set accuracy.
#Be sure to label the axes of your plots. Set the stopping criterion m=4 for these experiments.
def different_training_sizes(train_data, test_data):
    global row_number
    global accuracy
    global Set_of_nodes    
    global m
    Set_of_nodes = []
    small_data = pd.DataFrame()
    accuracy_graph = []
    del accuracy_graph[:]
    plt.clf() #Clears previous graphs.
    k = [0.05, 0.10, 0.20, 0.50, 1.00] 
    for index in k:
        del Set_of_nodes[:]
        row_number = 1
        accuracy = 0
        small_data = train_data.sample(frac=index, replace=True)
        Number_of_data_remaining = len(small_data)
        (positive, negative) = PositiveAndNegative(small_data)
        IG = Entropy(small_data)
        tree(small_data, m, Number_of_data_remaining, positive, negative, IG, -1)
        prediction(test_data)
        accuracy_graph.append(accuracy/len(test_data))
    print(accuracy_graph)
    plt.figure(1)
    plt.plot(k, accuracy_graph)   
    plt.xlabel("Training size (in %).")
    plt.ylabel("Test accuracy.")
    plt.title("Accuracy variation with different training sizes.")
    plt.savefig("Accuracy size - heart.png")


# In[28]:


#Q3. For this part, you will investigate how predictive accuracy varies as a function of tree size.
#For both of the data sets considered in Part 2, you should learn trees using the entire training set.
#Plot curves showing how test-set accuracy varies with the value m used in the stopping criteria.
#Show points for m = 2, 5, 10 and 20. Be sure to label the axes of your plots.
def leaf_size_variation(train_data, test_data):
    global row_number
    global accuracy
    global Set_of_nodes
    global m
    accuracy_graph = []
    del accuracy_graph[:]
    k = [2, 5, 10, 20]
    plt.clf() #Clears previous graphs.
    for index in k:
        del Set_of_nodes[:]
        row_number = 1
        accuracy = 0
        Number_of_data_remaining = len(train_data)
        (positive, negative) = PositiveAndNegative(train_data)
        IG = Entropy(train_data)
        tree(train_data, index, Number_of_data_remaining, positive, negative, IG, -1)
        prediction(test_data)
        accuracy_graph.append(accuracy/len(test_data))
    print(accuracy_graph)
    plt.figure(2)
    plt.plot(k, accuracy_graph)
    plt.xlabel("Limit of number of instances in leaf.")
    plt.ylabel("Test accuracy.")
    plt.title("Accuracy variation with leaf size.")
    plt.savefig("Accuracy leaf - heart.png")


# In[ ]:


#Main class.
if __name__ == "__main__":
    global row_number
    global accuracy
    global Set_of_nodes
    Set_of_nodes = []
    accuracy = 0
    del Set_of_nodes[:]
    Number_of_data_remaining = len(training_data)
    (positive, negative) = PositiveAndNegative(training_data)
    IG = Entropy(training_data)
    tree(training_data, m, Number_of_data_remaining, positive, negative, IG, -1)
    prediction(testing_data)
    #different_training_sizes(training_data, testing_data)
    #leaf_size_variation(training_data, testing_data)

