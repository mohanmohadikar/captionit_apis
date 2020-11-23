import pandas as pd
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
import pickle

# load model
model = pickle.load(open('model.pkl','rb'))
h5model = load_model(“model.h5”)
max_length = 32
tokenizer = pickle.load(open("tokenizer.p","rb"))
xception_model = Xception(include_top=False, pooling="avg")

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)
    print(data)

###imp start.

    

    photo = extract_features(img_path, xception_model)
    img = Image.open(img_path)

    description = generate_desc(model, tokenizer, photo, max_length)
    print("\n\n")
    print(description)

    ### imp end.
    

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

