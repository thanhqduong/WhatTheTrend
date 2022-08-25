from cProfile import label
from flask import Flask, request, render_template
import pandas as pd
import helpers

import io
import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


df = None
isSeasonal = None
m = None
# decompose_model = None

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/model_type', methods=['GET','POST'])
def get_data():
    data = request.form["data"]
    freq = request.form["freq"]
    data = helpers.strToList(data)
    global df
    df = helpers.listToDf(data, freq)
    img = 0
    # plot img df

    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Time-series")
    axis.set_xlabel("index")
    axis.set_ylabel("value")
    axis.grid()
    axis.plot(data, "ro-")
    
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')


    return render_template("model_type.html", img = pngImageB64String)

@app.route('/seasonal', methods=['GET','POST'])
def type1():
    choice = request.form['type']
    global df
    # global decompose_model
    decompose_model = helpers.decompose(df, choice)
    img = 0
    # return choice
    # plot img decompose_model.seasonal
    length = len(df.index)
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Seasonal Decomposition")
    axis.set_xlabel("index")
    axis.set_ylabel("value")
    axis.grid()
    axis.plot(range(1, length + 1), decompose_model.seasonal, "ro-")
    
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    return render_template("seasonal.html", img = pngImageB64String)
    

@app.route('/details', methods=['GET','POST'])
def type2():
    repeat = request.form['repeat']
    num_repeat = request.form['num_repeat']
    global isSeasonal
    global m
    isSeasonal = True
    m = int(num_repeat)
    if repeat == '0':
        isSeasonal = False
        m = 1
    global df
    length = len(df.index)
    return render_template("details.html", length = length)

@app.route('/result', methods=['GET','POST'])
def type3():
    train_length = int(request.form['train_test'])
    predict = int(request.form['predict'])
    global df
    # global decompose_model
    length = len(df.index)
    test_length = length - train_length
    total_pred = test_length + predict
    train, test = helpers.trainTestSplit(df, train_length)
    global isSeasonal
    global m
    model = helpers.gridSearch(df, seasonal=isSeasonal, m=m)

    model = helpers.fitModel(model, train)
    f = helpers.forecast(model, total_pred)
    valid_test = f[:test_length]
    forecast  = f[test_length:]
    # return df
    start_date = df.index[-1]
    date_list = pd.date_range(start=start_date, periods=(total_pred+1))
    # date_test = date_list[1: test_length + 1]
    # date_forecast = date_list[test_length + 1: ]
    img = 0
    # plot train test
    # return forecast.tolist()

    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Result")
    axis.set_xlabel("index")
    axis.set_ylabel("value")
    axis.grid()
    # return df['data'].tolist()
    axis.plot(range(1, length + 1), df['data'].tolist(), "bo-", label="value")
    axis.plot(range(1 + len(train), length + 1), valid_test, "go-", label='test')
    axis.plot(range(1 + length, 1 + length + predict), forecast, "ro-", label='predict')
    
    # axis.plot(date_test, valid_test)
    # axis.plot(date_forecast, forecast)
    
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    forecast = [round(e, 3) for e in forecast]

    return render_template("result.html", data = forecast, img = pngImageB64String)


if __name__ == '__main__':
    app.run(debug=True)