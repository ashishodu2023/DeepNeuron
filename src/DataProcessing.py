import os
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessing:
    def GetDigitData(self):
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
        file_path = "optdigits.tra"

        # Download the dataset
        if not os.path.exists(file_path):
            print("Downloading Optdigits dataset...")
            urllib.request.urlretrieve(url, file_path)
            print("Download complete.")

    # Function to preprocess the data and save it into input and target files
    def PreProcessSaveData(self):
        # Load the dataset
        data = np.loadtxt("C:\\Users\\Ashish\\PycharmProjects\\DeepNeuron\\optdigits.tra", delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save the preprocessed data into input and target files
        np.save("../train/X_train.npy", X_train)
        np.save("../train/y_train.npy", y_train)
        np.save("../test/X_test.npy", X_test)
        np.save("../test/y_test.npy", y_test)
        print('Files saved to train and test folder')



if __name__=='__main__':
    dp = DataProcessing()
    dp.GetDigitData()
    dp.PreProcessSaveData()