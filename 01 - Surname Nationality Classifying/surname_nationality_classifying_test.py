from surname_nationality_classifying import SurnameNationalityClassifyingModel

datasets_path = '../datasets/surnames_by_nationality/*.txt'
model_path = 'surname_nationality_model_state.pt'
model = SurnameNationalityClassifyingModel()

def train_model():
    model.load_dataset(datasets_path)
    model.train()
    model.save_model(model_path)

def load_saved_model():
    model.load_categories(datasets_path)
    model.load_saved_model(model_path)

def test_model():
    while True:
        name = input("Type a name: ")
        print('\n> %s' % name)
        predictions = model.predict(name)
        for loss, category in predictions:
            print('(%.3f) %s' % (loss, category))

#train_model()
load_saved_model()
test_model()