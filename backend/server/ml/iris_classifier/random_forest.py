# file backend/server/apps/ml/income_classifier/random_forest.py
import joblib
import pandas as pd

class RandomForestClassifier:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.model = joblib.load(path_to_artifacts + "random_forest.pkl")

    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        return input_data

    def predict(self, input_data):
        # print(input_data)
        # return self.model.predict_proba(input_data)
        return self.model.predict(input_data)

    def postprocessing(self, input_data):

        labels = ['setosa', 'versicolor', 'virginica']

        if input_data == 0:
            label = labels[0]
        if input_data == 1:
            label = labels[1]  
        if input_data == 2:
            label = labels[2]

        return {"probability": input_data, "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data) 
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction