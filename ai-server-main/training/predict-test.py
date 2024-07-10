import numpy as np
import pandas as pd
from keras.models import load_model

#model = load_model('server/model.h5')
model = load_model('./model.h5')

def getDiseaseNameDict():
    #originFile = pd.read_csv('server/dataset.csv', encoding='UTF-8')
    originFile = pd.read_csv('../data/dataset.csv', encoding='UTF-8')
    diseaseNameDict = {}
    processedData = originFile.to_numpy()

    count = 0

    for row in processedData:
        diseaseName = row[0]
        if diseaseNameDict.get(diseaseName) is None:
            diseaseNameDict[diseaseName] = count
            count += 1

    print(diseaseNameDict)

    return diseaseNameDict

diseaseNameDict = getDiseaseNameDict()

print(len(diseaseNameDict.keys()))

# 만성신장질환(매우높음)
#result = model.predict([[3, -1, -1, -1, 1, -1, -1, 6, 1.025, 3]], batch_size = 12)
#print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
#result = model.predict([[3, -1, -1, -1, 1, -1, -1, 5, 1.03, 3]], batch_size = 12)
#print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
#result = model.predict([[2, -1, -1, -1, 1, -1, -1, 6, 1.025, -1]], batch_size = 12)
#print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
#result = model.predict([[-1, 1, -1, -1, 1, -1, -1, 7, -1, -1]], batch_size = 12)
#print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
#result = model.predict([[1, -1, -1, -1, -1, -1, -1, 5, 1.03, 2]], batch_size = 12)
#print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
#result = model.predict([[-1, -1, -1, -1, -1, 1, 4, 5, 1.025, -1]], batch_size = 12)
#print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
#result = model.predict([[-1, -1, -1, -1, -1, -1, 4, 8, 1.1, -1]], batch_size = 12)
#print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])

# add 
result = model.predict([[3, 0, 0, 0, 1, 0, 0, 0, 0, 2]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[3, 0, 0, 0, 1, 0, 0, 0, 0, 3]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[3, 0, 0, 0, 2, 0, 0, 0, 0, 2]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[3, 0, 0, 0, 2, 0, 0, 0, 0, 3]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[2, 0, 0, 0, 1, 0, 0, 0, 0, 2]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[2, 0, 0, 0, 2, 0, 0, 0, 0, 3]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])

result = model.predict([[0, 1, 0, 0, 1, 0, 0, 0, 0, 3]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[0, 2, 0, 0, 1, 0, 0, 0, 0, 3]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[0, 1, 0, 0, 2, 0, 0, 0, 0, 3]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])


result = model.predict([[1, 0, 0, 0, 0, 0, 0, 0, 0, 2]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[1, 0, 0, 0, 0, 0, 0, 0, 0, 3]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[2, 0, 0, 0, 0, 0, 0, 0, 0, 2]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[2, 0, 0, 0, 0, 0, 0, 0, 0, 3]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[3, 0, 0, 0, 0, 0, 0, 0, 0, 3]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])

result = model.predict([[0, 0, 0, 0, 0, 1, 3, 0, 0, 0]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[0, 0, 0, 0, 0, 1, 4, 0, 0, 0]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[0, 0, 0, 0, 0, 0, 3, 0, 0, 0]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
result = model.predict([[0, 0, 0, 0, 0, 0, 4, 0, 0, 0]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])

result = model.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], batch_size = 12)
print(list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))])
