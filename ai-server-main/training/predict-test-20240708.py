import numpy as np
import pandas as pd
from keras.models import load_model

import sys, os 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from server.judge_level import judgeLevel

model = load_model('../model/model-20240708.h5')

def getDiseaseNameDict(filePath):
    originFile = pd.read_csv(filePath, encoding='UTF-8')
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

def testAll(filePath):
    originFile = pd.read_csv(filePath, encoding='UTF-8')
    modifiedFile = originFile.replace(['-', 'ㅡ', None], 0)
    modifiedFile = modifiedFile.replace(['+'], '1')
    modifiedFile = modifiedFile.replace(['+'], '1')
    modifiedFile = modifiedFile.replace([r'(\d)(\+)'], r'\1', regex=True)
    modifiedFile = modifiedFile.replace([r'(^>)(\d)'], r'\2', regex=True)
    modifiedFile = modifiedFile.fillna(0)
    # modifiedFile['ph'] = 0
    # modifiedFile['sg'] = 0
    numpyd = modifiedFile[['label', 'blood','bilirubin','urobilinogen','ketones','protein', 'nitrite','glucose','ph','sg','leukocytes']].to_numpy()
    # numpyd = modifiedFile[['label', 'blood','bilirubin','urobilinogen','ketones','protein', 'nitrite','glucose','leukocytes']].to_numpy()
    failCount = 0
    partialFailCount = 0
    for row in numpyd:
        data = list(row[1:].astype(float))
        orgLabel = row[0]
        result = model.predict([data], batch_size = 12)
        predLabel = list(diseaseNameDict.keys())[list(diseaseNameDict.values()).index(np.argmax(result))]
        level = judgeLevel(predLabel, data)

        print(f"Data = {data}, 학습 라벨 = {orgLabel}, 예측 라벨 = {predLabel}, 레벨 = {level}")
        if orgLabel != predLabel:
            diseaseNames = [value for value in orgLabel.split(",") if value in predLabel.split(",")]
            print(f"    ===> Not matched, Paritial Matched Diseases = {diseaseNames}")
            if len(diseaseNames) == 0:
                partialFailCount += 1
            input()
            failCount += 1
    print(f"예측 실패 횟수 = {failCount}, 실패율 = {100 * failCount / len(numpyd)}%, 성공률 = {100 - 100 * failCount / len(numpyd)}%")
    print(f"부분 실패 횟수 = {partialFailCount}, 부분실패율 = {100 * partialFailCount / len(numpyd)}%")

diseaseNameDict = getDiseaseNameDict('../data/dataset-20240708.csv')
print(len(diseaseNameDict.keys()))

testAll('../data/dataset-20240708.csv')
