#!/usr/bin/env python
# coding: utf-8

# ### Prayatul Matrix for direct comparison of Machine  Learning algorithms                                                          
#   Source codes demo version 1.0                                                                      
#                                                                                                   
#   Developed in Python 3                                                                   
#                                                                                                      
#   Author and programmer: Anupam Biswas                                                          
#                                                                                                     
#         e-Mail:    anupam@cse.nits.ac.in                                                
#                                                                                                                                    Homepage: http://cs.nits.ac.in/anupam/    
#                                                                                                                                  
#                                                                                                                                    

# In[27]:


import numpy as np


# ### Prayatul Matrix Generation
# The generatePrayatulMatrix function takes three inputs
#  1) G : Ground truth labels of all samples in the dataset
#  2) P : Predicted labels obtained with primary algorithm
#  3) Q : Predicted labels obtained with alternative algorithm

# In[28]:


def generatePrayatulMatrix( G,P,Q ):
    D = np.array([[0,0], [0,0]])
    for i in range(len(G)):        
        if P[i]==G[i] and Q[i]==G[i]:            
            D[0][0] +=1
        elif P[i]==G[i] and Q[i]!=G[i]:
            D[0][1] +=1            
        elif P[i]!=G[i] and Q[i]==G[i]:
            D[1][0] +=1             
        elif P[i]!=G[i] and Q[i]!=G[i]:
            D[1][1] +=1   
    return D


# ### Scores based on Prayatul Matrix
# The getScore function takes a prayatul matrix as input and return five scores

# In[29]:


def getScores(D):
    BR=D[0][0]
    RW=D[0][1]
    WR=D[1][0]
    BW=D[1][1]
    k=0.0001  # k is a very small quantity added to the denominator to avoid x/0 situation
    if RW==0 and WR==0:
        sigmaC=(RW-WR)/(RW+WR+k)
    else:
        sigmaC=(RW-WR)/(RW+WR)
    if BR==0 and BW==0:
        alpha=(BR-BW)/(BR+BW+k)
    else:
        alpha=(BR-BW)/(BR+BW)
    if BR==0 and RW==0 and WR==0:
        xiC=(BR+RW)/(BR+RW+WR+k)
        xiE=(BR+RW-WR)/(BR+RW+WR+k)
    else:
        xiC=(BR+RW)/(BR+RW+WR)
        xiE=(BR+RW-WR)/(BR+RW+WR)    
    phiE=(BR+RW-WR)/(BR+RW+WR+BW)
    return sigmaC, alpha, xiC, xiE, phiE


# In[30]:


# Ground truth labels for all samples
G=[1,1,1,0,0,0,1,0,0,1,1,0,1,1,0,1,1,1,0,1]

# Predicted labels for all samples obtained with primary ML algorithm, whose performance is to be compared
P=[1,1,1,0,1,0,1,1,0,0,1,0,1,0,0,1,1,1,1,1]

# Predicted labels for all samples obtained with alternative ML algorithm with whom the primary algorithm is to be compared
Q=[1,0,1,0,1,1,0,1,0,1,0,1,1,1,0,0,1,0,0,1]


D=generatePrayatulMatrix(G,P,Q)
# Display Prayatul Matrix D
print(D)

sigmaC, alpha, xiC, xiE, phiE = getScores(D)
# Display Scores obatained based on the Prayatul Matrix prepared for two ML algorithms 
print(sigmaC, alpha, xiC, xiE, phiE)

