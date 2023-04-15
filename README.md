**MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles.**
Modern automobiles, particularly connected and autonomous vehicles, contain several electronic control units that are linked via intra-vehicle networks to implement various functionalities and conduct actions. In order to communicate with other vehicles, infrastructure, and smart gadgets, modern automobiles are also connected to external networks via vehicle-to-everything technology. Modern cars are becoming more useful and connected, but because to their broad attack surfaces, this also makes them more susceptible to cyberattacks that target both internal and external networks. A lot of effort has gone into creating intrusion detection systems (IDSs) that use machine learning techniques to identify hostile cyber-attacks in order to secure vehicle networks.

In order to identify both known and unidentified assaults on vehicular networks, a multi-tiered hybrid IDS that combines a signature-based IDS and an anomaly-based IDS is proposed in this research. The vulnerabilities of intra-vehicle and external networks are also examined. The CICIDS2017 dataset and the CAN-intrusion-dataset, which represent the data from inside and outside of vehicles, respectively, show that the suggested system can accurately detect a variety of known attacks.

The proposed MTH-IDS framework comprises of four tiers of learning models in addition to two conventional ML stages (data pre-processing and feature engineering): Four tree-based supervised learners - decision tree (DT), random forest (RF), extra trees (ET), and extreme gradient boosting (XGBoost) — used as multi-class classifiers for known attack detection.
A stacking ensemble model and a Bayesian optimization with tree Parzen estimator (BO-TPE) method for supervised learner optimization. 
A cluster labeling (CL)k-means used as an unsupervised learner for zero-day attack detection; 
Two biased classifiers and a Bayesian optimization with Gaussian process (BO-GP) method for unsupervised learner optimization.

**Required Environments and Softwares :**

- [Anaconda Navigator 3 ](https://www.anaconda.com/products/distribution/ "Anaconda Navigator 3"). 
   - [Jupyter version 6.3.0] ( In anaconda navigator you can launch the jupyter nodebook ).  
- [Python 3]( https://www.python.org/downloads/ "Python 3"). 
- [Google Collab]( https://colab.research.google.com/ ). 

**Datasets :**
- [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html "CICIDS2017"). 
- [CAN_Intrusion_Dataset](https://www.dropbox.com/sh/b0asam3c45k607a/AAChCVjjIG5v4popd1FdryKSa?dl=0/ "CAN Intrusion Data Set").  
- [AWS IOT Fleetwise](https://us-east-1.console.aws.amazon.com/iotfleetwise/home?region=us-east-1# "AWS IOT Fleetwise"). 

**Libraries used :**

Libraries/packages used for data preprocessing and feature engineering:

**NumPy :**
“import numpy as np”

1. NumPy is the fundamental package needed for scientific computing with Python. This package
contains:
• A powerful N-dimensional array object
• sophisticated (broadcasting) functions
• basic linear algebra functions
• basic Fourier transforms
• sophisticated random number capabilities
NumPy is a successor for two earlier scientific Python libraries: Numeric and Numarray.

**Pandas :**

“import pandas as pd”
1. Python's Pandas package is used to manipulate data sets. It offers tools for data exploration, cleaning, analysis, and manipulation. Both "Panel Data" and "Python Data Analysis" are mentioned in the term "Pandas". With the aid of Pandas, we can examine large data sets and draw conclusions based on statistical principles.

**Seaborn :**
“import seaborn as sns”
A package called Seaborn uses Matplotlib as its foundation to plot graphs. In order to see random distributions, it will be used.

**matplotlib.pyplot :**

“import matplotlib.pyplot as plt”

A MATLAB-like interface is offered by the Matplotlib plugin Pyplot. With the ability to use Python and the benefit of being free and open-source, Matplotlib is made to be just as usable as MATLAB.

**From skyline.preprocessing import LabelEncoder :**

“from skyline.preprocessing import LabelEncoder”

Target labels with values between 0 and n classes-1 should be encoded.
Instead of encoding the input X, this transformer should be used to encode the target values, or y.

**From sklearn.model_selection import train_test_split :**

“from sklearn.model_selection import train_test_split”

Python's train test split method divides arrays or matrices into random subsets for the train and test sets of data, respectively.

**Commands to run the code**

1. Open the Anaconda Navigator and then launch the Jupyter Notebook.
2. Now upload the source code file and dataset.
3. Click on run cells.

**Execution flow**

 **<p align="center">Figure 7: Flow Chart</p>**
<p align="center">
<img src="https://github.com/Shireesha21/MTH_IDS_IoV/blob/main/MicrosoftTeams-image%20(5).png" width="500" />
</p>

 
**Issues experiencing and solutions :**

**Minor Issues and their solutions :**
1. Installation 

   Re-installed upgraded versions
2. Dataset compatability with tools

   Switched to Google collab as it accepts large datasets.
3. Data 

   Implemented Data preprocessing to encounter the missing/invalid data.

**Major Issues and their solutions**
1. Choosing appropriate service for creating the network environment.

   AWS IOT Fleetwise provides over-the-air (OTA) update and automation functionalities.
2. Configuring the dbc files
3. Campaigning the Vehicle using AWS IOT Fleetwise 

 
   
**FeedBack and Suggestions**
****

Feedback

To improve the accuracy of identifying vulnerabilities and attacks on our vehicle dataset using hybrid-based algorithms, we can preprocess the expanded dataset through techniques such as data cleaning, normalization, feature extraction, and dimensionality reduction. After applying these techniques, the algorithms can be reapplied to obtain accurate results and insights.

Suggestions

Instead of using AWS, can use Network Simulators like GNS3. Create a topology with nodes and communicate between them and Use Wireshark and npcap to capture the data. Perform an attack and apply ML algorithms.


References:

https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning

https://docs.aws.amazon.com/pdfs/iot-fleetwise/latest/developerguide/iot-fleetwise-guide.pdf#process-visualize-data
