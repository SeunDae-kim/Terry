import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import adam_v2
from keras.utils import to_categorical

def createDataset(filePath):
    originFile = pd.read_csv(filePath, encoding='UTF-8')
    modifiedFile = originFile.replace(['-', 'ㅡ', None], 0)
    modifiedFile = modifiedFile.replace(['+'], '1')
    modifiedFile = modifiedFile.replace(['+'], '1')
    modifiedFile = modifiedFile.replace([r'(\d)(\+)'], r'\1', regex=True)
    modifiedFile = modifiedFile.replace([r'(^>)(\d)'], r'\2', regex=True)
    modifiedFile = modifiedFile.fillna(0)
    # modifiedFile['ph'] = 0
    # modifiedFile['sg'] = 0
    numpyd = modifiedFile[['label','blood','bilirubin','urobilinogen','ketones','protein', 'nitrite','glucose','ph','sg','leukocytes']].to_numpy()
    # numpyd = modifiedFile[['label','blood','bilirubin','urobilinogen','ketones','protein', 'nitrite','glucose','leukocytes']].to_numpy()
    numpyd = np.tile(numpyd, (3, 1))
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


def trainModel(datasetDict, checkpoint, epoch, batch_size, dropout):
    inputCount = datasetDict['xTrainDataset'].shape[1]
    outputCount = datasetDict['yTrainDataset'].shape[1]

    model = Sequential()
    model.add(Dense(10, input_dim=inputCount, activation='relu', name='layer1'))
    model.add(Dropout(rate=dropout))
    model.add(Dense(64, activation='relu', name='layer2'))
    model.add(Dense(128, activation='relu', name='layer3'))
    model.add(Dropout(rate=dropout))
    model.add(Dense(128, activation='relu', name='layer4'))
    model.add(Dense(outputCount, activation='softmax', name="layer5"))
    model.compile(optimizer=adam_v2.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    hist = model.fit(datasetDict['xTrainDataset'],
                     datasetDict['yTrainDataset'],
                     epochs=epoch,
                     batch_size=batch_size,
                     validation_data=(datasetDict['xTestDataset'], datasetDict['yTestDataset']),
                     callbacks=[checkpoint])

    return {'model': model, 'history': hist}


import matplotlib.pyplot as plt

def run(filePath, checkpoint, epoch, batch_size, dropout):
    dataset = createDataset(filePath)
    processedDataset = processDataset(dataset)
    result = trainModel(processedDataset, checkpoint, epoch, batch_size, dropout)
    history = result['history'].history

    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['loss'])
    plt.title('training')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history['val_accuracy'])
    plt.plot(history['val_loss'])
    plt.title('validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()

# originFilePath = '../data/dataset-new.csv'
originFilePath = '../data/dataset-new-2024.csv'

EPOCH = 200

# batch_sizes = [30, 40, 50, 60, 70, 80, 90, 100]
# batch_sizes = [70, 80, 90, 100]
BATCH_SIZE = 50

# for BATCH_SIZE in batch_sizes:
# filename = '../model/model-new.h5'.format(EPOCH, BATCH_SIZE)
filename = '../model/model-new-2024.h5'.format(EPOCH, BATCH_SIZE)
checkpoint = ModelCheckpoint(filename,
                         monitor='val_loss',
                         verbose=1,
                         save_best_only=True,
                         mode='auto')

print("*"*80)
print(f"***** RUNNING - BATCH SIZE = {BATCH_SIZE}")
print("*"*80)
run(originFilePath, checkpoint, EPOCH, BATCH_SIZE, 0.4)
