from keras.models import model_from_json
from keras.models import load_model

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json.file_write(model_json)

model.save_weights("model.h5")
print("saved model to disk")
