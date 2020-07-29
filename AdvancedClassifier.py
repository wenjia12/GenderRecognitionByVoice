import numpy as np
import matplotlib.pyplot as plt
import numpy as np    
import os
import pandas as pd

voice_data=pd.read_csv('voice.csv')
male=voice_data.iloc[:1583,:]
male_x1=male['IQR']
male_x2=male['meanfun']

female=voice_data.iloc[1584:,:]
female_x1=female['IQR']
female_x2=female['meanfun']

plt.figure()
plt.scatter(male_x1,male_x2,c='b',alpha=0.5,label='male')
plt.scatter(female_x1,female_x2,c='r',alpha=0.5,marker="p",label='female')
plt.xlabel('IQR')
plt.ylabel('meanfun')
plt.legend(loc='upper right')    
plt.show()
