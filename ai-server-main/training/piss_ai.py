import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import adam_v2
from keras.utils import to_categorical

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EPOCH = 100
BATCH_SIZE = 12

#filename = 'model-new.h5'.format(EPOCH, BATCH_SIZE)
filename = '/home/wisoft/Optosta/ai-server/model/model.h5'.format(EPOCH, BATCH_SIZE)
# originFilePath = '../data/dataset.csv'
#originFilePath = '../data/dataset-new.csv'
originFilePath = '/home/wisoft/Optosta/ai-server/data/test.csv'

checkpoint = ModelCheckpoint(filename,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')


def createDataset(filePath):
    originFile = pd.read_csv(filePath, encoding='UTF-8')
    modifiedFile = originFile.replace(['-', 'ㅡ', None], 0)
    modifiedFile = modifiedFile.replace(['+'], '1')
    modifiedFile = modifiedFile.replace([r'(\d)(\+)'], r'\1', regex=True)
    print(modifiedFile.to_numpy())

    numpyd = modifiedFile[['label','blood','bilirubin','urobilinogen','ketones','protein', 'nitrite','glucose','ph','sg','leukocytes']].to_numpy()
    print(numpyd)

    return numpyd


def processDataset(dataset):
    diseaseNameDict = {}
    processedData = dataset

    count = 0

    for row in dataset:
        diseaseName = row[0]
        if (diseaseNameDict.get(diseaseName) is None):
            diseaseNameDict[diseaseName] = count
            count += 1

    for (name, value) in diseaseNameDict.items():
        processedData = np.where(processedData == name, value, processedData)

    np.random.shuffle(processedData)
    processedData = processedData.astype(float)

    xData = processedData[0:, 1:]
    yData = processedData[0:, 0:1]

    trainCount = int(len(xData) * 0.7)

    xTrainData = xData[:trainCount, :]
    yTrainData = yData[:trainCount, :]

    xTestData = xData[trainCount:, :]
    yTestData = yData[trainCount:, :]

    yTrainHotData = to_categorical(yTrainData, num_classes=len(diseaseNameDict))
    yTestHotData = to_categorical(yTestData, num_classes=len(diseaseNameDict))

    return {'xTrainDataset': xTrainData, 'yTrainDataset': yTrainHotData, 'xTestDataset': xTestData,
            'yTestDataset': yTestHotData}


def trainModel(datasetDict):
    inputCount = datasetDict['xTrainDataset'].shape[1]
    outputCount = datasetDict['yTrainDataset'].shape[1]

    model = Sequential()
    model.add(Dense(64, input_dim=inputCount, activation='relu', name='layer1'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(128, activation='relu', name='layer2'))
    model.add(Dense(128, activation='relu', name='layer3'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(64, activation='relu', name='layer4'))
    model.add(Dense(outputCount, activation='softmax', name="layer5"))
    model.compile(optimizer=adam_v2.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    hist = model.fit(datasetDict['xTrainDataset'],
                     datasetDict['yTrainDataset'],
                     epochs=EPOCH,
                     batch_size=BATCH_SIZE,
                     validation_data=(datasetDict['xTestDataset'], datasetDict['yTestDataset']),
                     callbacks=[checkpoint])

    return {'model': model, 'history': hist}


def run(filePath):
    dataset = createDataset(filePath)
#     dataset = createDataset(originFilePath)
    processedDataset = processDataset(dataset)
    trainModel(processedDataset)

#run(originFilePath)
