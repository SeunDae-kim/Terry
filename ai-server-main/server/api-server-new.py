from flask import Flask, request, jsonify, make_response
import numpy as np
import pandas as pd
import requests
from keras.models import load_model
import sys, os 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from training.piss_ai_new import run


class PredictService:
    def __init__(self):
        self.model = load_model('../model/model-new.h5')
        # self.model = load_model('/home/wisoft/Optosta/ai-server/model/model-new.h5')
        self.diseaseNameDict1 = {}
        originFile = pd.read_csv('../data/dataset-new.csv', encoding='UTF-8')
        # originFile = pd.read_csv('/home/wisoft/Optosta/ai-server/data/dataset-new.csv', encoding='UTF-8')
        self.diseaseNameDict = {}
        processedData = originFile.to_numpy()

        count = 0

        for row in processedData:
            diseaseName = row[0]
            if self.diseaseNameDict.get(diseaseName) is None:
                self.diseaseNameDict[diseaseName] = count
                count += 1

        print(self.diseaseNameDict)

    def predict(self, predictRequest):
        result = self.model.predict([predictRequest.getData()], batch_size=50)
        print(predictRequest.getData())
        print(result)
        return self.getDiseaseName(result)

    def getDiseaseName(self, result):
        keys = self.diseaseNameDict.keys()
        values = self.diseaseNameDict.values()
        diseaseIndex = np.argmax(result)
        return list(keys)[list(values).index(diseaseIndex)]


class PredictRequest:
    def __init__(self, json):
        self.blood = int(json['blood'])  # 잠혈
        self.bilirubin = int(json['bilirubin'])  # 빌리루빈
        self.urobilinogen = int(json['urobilinogen'])  # 우로빌리로겐
        self.ketones = int(json['ketones'])  # 케톤체
        self.protein = int(json['protein'])  # 단백질
        self.nitrite = int(json['nitrite'])  # 아질산염
        self.glucose = int(json['glucose'])  # 포도당
        # self.ph = int(json['ph'])  # pH
        # self.sg = float(json['sg'])  # 비중S.G
        self.leukocytes = int(json['leukocytes'])  # 백혈구

    def getData(self):
        return [
            self.blood,
            self.bilirubin,
            self.urobilinogen,
            self.ketones,
            self.protein,
            self.nitrite,
            self.glucose,
            # self.ph,
            # self.sg,
            self.leukocytes
        ]


app = Flask(__name__)
predictService = PredictService()


@app.route('/', methods=['GET'])
def index():
    return 'WiSoft.io'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request.get_json())
        result = predictService.predict(PredictRequest(request.get_json()))
        return make_response(jsonify({'result': result}), 200)
    except Exception as e:
        print(e)
        return make_response(jsonify({'result': 'ERROR'}), 500)


@app.route('/training', methods=["GET"])
def training():
    try:
        response = requests.get("http://192.168.20.241:8080/api/trainings")
        json_to_csv(response.json())
        run("/home/wisoft/Optosta/ai-server/data/test.csv")
        return make_response(jsonify({}), 200)
    except Exception as e:
        print(e)
        return make_response(jsonify({'result': 'ERROR'}), 500)


@app.route('/new-predict', methods=['POST'])
def new_predict():
    try:
        data = request.get_json()
        dataFrame = pd.DataFrame(data, index=[0])
        dataFrame = dataFrame.replace(['-', 'ㅡ', None], 0)
        dataFrame = dataFrame.replace(['+'], '1')
        dataFrame = dataFrame.replace([r'(\d)(\+)'], r'\1', regex=True)
        dataFrame = dataFrame.replace([r'(^>)(\d)'], r'\2', regex=True)
        dataFrame = dataFrame.fillna(0)
        json = dataFrame.loc[0].to_dict()
        result = predictService.predict(PredictRequest(json))
        print(result);
        return make_response(jsonify({'result': result}), 200)
    except Exception as e:
        print(e)
        return make_response(jsonify({'result': 'ERROR'}), 500)

def json_to_csv(json):
    
    print(json)
    df = pd.DataFrame(json)
    print(df)
    df.to_csv("/home/wisoft/Optosta/ai-server/data/test.csv", index=False)

if __name__ == '__main__':
    app.run('0.0.0.0', 10005)
