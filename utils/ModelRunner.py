from Surname_Nationality_Classifying.surname_nationality_classifying import *

datasets_path = '../datasets/surnames_by_nationality/*.txt'
model_path = 'Surname_Nationality_Classifying/surname_nationality_model_state.pt'
model = SurnameNationalityClassifyingModel()

class ModelRunner():
    def __init__(self):
        model.load_saved_model(model_path)

    def predict(self, name):
        predictions = model.predict(name)
        return predictions 