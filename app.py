from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM,Input
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
#from polygon import RESTClient
import matplotlib.pyplot as plt
import io
import base64
import requests
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize global model variable
model = None
scaler = MinMaxScaler(feature_range=(0, 1))

# Load the trained model
def load_trained_model():
    global model
    model = Sequential()
    model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.load_weights("model.h5")

load_trained_model()

# Function to fetch the latest Bitcoin prices automatically (example implementation)
def fetch_latest_prices(time_step):
    api_key = '8W7n3sVbOnWYnmCIRpUAdhIwWib1fQ8u'
    api_url = 'https://api.polygon.io/v2/aggs/ticker/X:BTCUSD/prev'
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    
    # Request the latest Bitcoin prices
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Check if the request was successful
        data = response.json()
        
        # Check if data is returned properly
        if 'results' not in data:
            raise ValueError("No results found in API response")
        
        # Extract the prices from the API response
        prices = [result['c'] for result in data['results']]
        
        # Ensure we return only as many prices as requested by time_step
        return prices[-time_step:]
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []
    except ValueError as e:
        print(f"Value error: {e}")
        return []
    
# Function to plot historical data
def plot_historical_data(df):
    fig = px.line(df, x=df.index, y='closing_price', labels={'x': 'Date', 'y': 'Price'}, title="Historical Bitcoin Prices")
    fig.update_layout(showlegend=False, plot_bgcolor='white', font_size=15, font_color='black')
    return fig

# Function to plot prediction results
def plot_prediction_results(last_original_days_value, next_predicted_days_value):
    fig = px.line(x=np.arange(len(last_original_days_value) + len(next_predicted_days_value)),
                  y=np.concatenate((last_original_days_value, next_predicted_days_value)),
                  labels={'x': 'Days', 'y': 'Price'},
                  title="Bitcoin Price Prediction")
    fig.update_layout(showlegend=False, plot_bgcolor='white', font_size=15, font_color='black')
    return fig

# ARIMA Prediction
def arima_prediction(df, time_step, pred_days, conf_interval):
    df = df.rename(columns={'closing_price': 'y'})
    model = ARIMA(df['y'], order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    
    # Forecast with prediction intervals
    forecast, stderr, conf_int = model_fit.forecast(steps=pred_days, alpha=1 - conf_interval / 100)
    
    return forecast, conf_int

@app.route('/', methods=['GET', 'POST'])
def index():
    historical_graph_html = None
    prediction_graph_html = None

    if request.method == 'POST':
        global model  # Ensure the model is accessible
        time_step = int(request.form['time_step'])
        pred_days = int(request.form['pred_days'])
        model_choice = request.form.get('model_choice', 'lstm')
        auto_fetch = 'auto_fetch' in request.form
        file = request.files.get('csv_file')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        if file and file.filename.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            if start_date and end_date:
                df_new = df.loc[start_date:end_date]

            # Plot historical data
            historical_fig = plot_historical_data(df_new)
            historical_graph_html = historical_fig.to_html(full_html=False)

            # Extract last time_step number of closing prices
            last_close_prices = df['closing_price'].values[-time_step:].tolist()
            print("Last Close Prices:", last_close_prices)

            if model_choice == 'lstm':
                # Scale and prepare data
                last_close_prices = np.array(last_close_prices).reshape(-1, 1)
                scaler.fit(last_close_prices)
                scaled_data = scaler.transform(last_close_prices)
                temp_input = list(scaled_data[-time_step:].reshape(1, -1)[0])
                print("Scaled Data:", scaled_data)
                print("Temp Input for Prediction:", temp_input)

                # Predict future prices
                lst_output = []
                for i in range(pred_days):
                    x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
                    yhat = model.predict(x_input, verbose=0)
                    temp_input.append(yhat[0][0])
                    lst_output.append(yhat[0][0])
                    #print(yhat.shape)
                
                print("Model Predictions:", lst_output)
                print("Shape of predicted list:",len(lst_output))

                # Inverse transform the predicted prices
                try:
                    predicted_prices = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten()
                except ValueError as e:
                    print("Inverse Transform Error:", e)
                    predicted_prices = []

            elif model_choice == 'arima':
                # ARIMA prediction
                predicted_prices, _ = arima_prediction(df, time_step, pred_days, conf_interval=95)

            # Prepare data for plotting
            last_original_days_value = df['closing_price'].values[-time_step:]
            next_predicted_days_value = predicted_prices

            # Plot results
            prediction_fig = plot_prediction_results(last_original_days_value, next_predicted_days_value)
            prediction_graph_html = prediction_fig.to_html(full_html=False)

        else:
            if auto_fetch:
                last_close_prices = fetch_latest_prices(time_step)
            else:
                last_close_prices = [float(x) for x in request.form['last_close_prices'].split(',')]
            print("Shape of manually enter input:",len(last_close_prices))
            # Scale and prepare data
            last_close_prices = np.array(last_close_prices).reshape(-1, 1)
            scaler.fit(last_close_prices)
            scaled_data = scaler.transform(last_close_prices)
            temp_input = list(scaled_data[-time_step:].reshape(1, -1)[0])

            # Predict future prices
            lst_output = []
            for i in range(pred_days):
                x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
                yhat = model.predict(x_input, verbose=0)
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])

            print("Model Predictions:", lst_output)

            # Inverse transform the predicted prices
            try:
                predicted_prices = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten()
                print("Predicted prices are:",predicted_prices)
                print("Predicted output shape is:",len(predicted_prices))
            except ValueError as e:
                print("Inverse Transform Error:", e)
                predicted_prices = []

            # Prepare data for plotting
            last_original_days_value = last_close_prices.flatten()
            next_predicted_days_value = predicted_prices

            # Plot results
            prediction_fig = plot_prediction_results(last_original_days_value, next_predicted_days_value)
            prediction_graph_html = prediction_fig.to_html(full_html=False)

    return render_template('index.html', historical_graph_html=historical_graph_html, graph_html=prediction_graph_html)

@app.route('/train', methods=['GET', 'POST'])
def train():
    global model  # Ensure the model is accessible

    if request.method == 'POST':
        file = request.files.get('train_csv_file')
        time_step = request.form.get('time_step')  # Use .get() to avoid KeyError

        if not time_step:
            return "Time step is required", 400

        try:
            time_step = int(time_step)
        except ValueError:
            return "Invalid time step value", 400

        if file and file.filename.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # Prepare data
            data = df['closing_price'].values
            data = data.reshape(-1, 1)
            scaled_data = scaler.fit_transform(data)

            # Define and train the model
            model = Sequential()
            model.add(Input(shape=(time_step, 1)))
            model.add(LSTM(50, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            X_train = []
            y_train = []
            for i in range(len(scaled_data) - time_step):
                X_train.append(scaled_data[i:i + time_step])
                y_train.append(scaled_data[i + time_step])
            X_train, y_train = np.array(X_train), np.array(y_train)

            model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=2)
            model.save('trained_model.h5')

        return redirect(url_for('index'))

    return render_template('train.html')


if __name__ == "__main__":
    app.run(debug=True)
