import joblib
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from joblib import dump
from flask_cors import CORS
from datetime import datetime


app = Flask(__name__)
CORS(app, supports_credentials=True)

# Data directory prefix for Docker volume
DATA_DIR = '/data/'

@app.route('/')
def hello_world():
    return jsonify({"message": "Hello, World!"})

# Data preview
@app.route('/preview', methods=['GET'])
def get_data():
    df = pd.read_excel(DATA_DIR + 'Data-building-energy-consumption.xlsx', decimal=',')


    # Convert the 'Time' column to string to make it JSON serializable
    df['Time'] = df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Prepare the data for eCharts
    chart_data = {
        "time": df['Time'].tolist(),
        "values": df['building 41'].tolist()
    }

    # Send the data
    return jsonify(chart_data)

@app.route('/filter-data', methods=['POST'])
def filter_data():
    # Load the data
    df = pd.read_excel(DATA_DIR + 'Data-building-energy-consumption.xlsx', decimal=',')

    # Get date range from POST request
    data = request.get_json()
    start_date = datetime.strptime(data['startDate'], '%Y-%m-%dT%H:%M:%S.%fZ')
    end_date = datetime.strptime(data['endDate'], '%Y-%m-%dT%H:%M:%S.%fZ')

    # Ensure 'Time' column is in datetime format for filtering
    df['Time'] = pd.to_datetime(df['Time'])

    # Filter the DataFrame based on the date range
    filtered_df = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)].copy()

    # Convert 'Time' back to string for JSON serialization
    filtered_df['Time'] = filtered_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Prepare the filtered data for response
    chart_data = {
        "time": filtered_df['Time'].tolist(),
        "values": filtered_df['building 41'].tolist()
    }

    return jsonify(chart_data)

# Data cleaning
@app.route('/cleaning', methods=['GET'])
def get_cleaning():
    # Loading the datasets
    Building = pd.read_excel(DATA_DIR + 'Data-Building-energy-consumption.xlsx')
    Building = Building.set_index("Time")

    Energy = pd.read_excel(DATA_DIR + 'Data-Weather.xlsx')
    Energy = Energy.set_index("Time")

    # Concatenating the datasets
    df = pd.concat([Building, Energy], axis=1)

    # Checking missing data
    missing_data = df.isna().sum().to_dict()

    df.to_csv(DATA_DIR + 'concatenated_data.csv', index=True)

    # Creating a response object with only the first 10 rows of the concatenated dataset
    response = {
        "concatenated_data_first_10_rows": df.head(10).to_json(),
        "missing_data": missing_data
    }

    return jsonify(response)

# Data correlation
@app.route('/all-correlation', methods=['GET'])
def get_all_correlation():
    df = pd.read_csv(DATA_DIR + 'concatenated_data.csv')  # Load your data

    # Exclude non-numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=[float, int])

    correlation_matrix = numeric_df.corr().to_json()
    return jsonify(correlation_matrix)

@app.route('/correlation', methods=['POST'])
def get_correlation():
    # Load your data
    df = pd.read_csv(DATA_DIR + 'concatenated_data.csv')

    # Retrieve column names from POST request
    data = request.get_json()
    columns = data.get('columns')

    if not columns:
        return jsonify({"error": "No columns specified"}), 400

    # Check if the specified columns exist in the DataFrame
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        return jsonify({"error": "Columns not found in dataset: " + ", ".join(missing_columns)}), 404

    # Filter the DataFrame to include only the specified columns
    filtered_df = df[columns]
    num_filtered_df = filtered_df.select_dtypes(include=[float, int])

    # Calculate correlation matrix
    correlation_matrix = num_filtered_df.corr().to_json()

    return jsonify(correlation_matrix)


@app.route('/get-weekly-data', methods=['GET'])
def get_weekly_data():
    df = pd.read_csv(DATA_DIR + 'concatenated_data.csv', index_col='Time', parse_dates=True)  # Adjust as necessary
    df_sum_weekly = df['building 41'].resample('W').mean()
    df_feature1 = df["Temp"].resample("W").mean()
    df_feature2 = df["U"].resample("W").mean()

    response = {
        "energy": df_sum_weekly.tolist(),
        "temperature": df_feature1.tolist(),
        "humidity": df_feature2.tolist(),
        "dates": df_sum_weekly.index.strftime('%Y-%m-%d').tolist()
    }

    return jsonify(response)


@app.route('/train-model', methods=['GET'])
def train_model():
    # Load your data (adjust paths and loading method as needed)
    df = pd.read_csv(DATA_DIR + 'concatenated_data.csv', parse_dates=['Time'])
    df.set_index('Time', inplace=True)

    # Extracting the energy consumption data
    energy = np.array(df["building 41"])

    # Reduce number of features
    knmi_updated = df.loc[:, ~df.columns.isin(["TD", "U", "DR", "FX"])]


    X = knmi_updated  # No need to reshape if it's already in the correct format
    y = energy  # Use the 'energy' array as the target


    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Save the test sets to CSV files
    X_test.to_csv(DATA_DIR + 'X_test.csv', index=False)
    pd.DataFrame(y_test, columns=['y_test']).to_csv(DATA_DIR + 'y_test.csv', index=False)

    # Train the model
    model = RandomForestRegressor(max_depth=30, random_state=0)
    model.fit(X_train, y_train)

    # Predicting on the training data
    Predicted_Train = model.predict(X_train)

    # Evaluating the model
    r2 = r2_score(y_train, Predicted_Train)
    mse = mean_squared_error(y_train, Predicted_Train)

    # Save the trained model to a file
    dump(model, DATA_DIR + 'trained_random_forest_model.joblib')

    # Return the performance metrics
    return jsonify({
        "R2_Score": r2,
        "Mean_Squared_Error": mse
    })

@app.route('/evaluate-model', methods=['GET'])
def evaluate_model():
    # Load the test data from CSV files
    X_test = pd.read_csv(DATA_DIR + 'X_test.csv')
    y_test = pd.read_csv(DATA_DIR + 'y_test.csv')

    #load model joblib
    model = joblib.load('trained_random_forest_model.joblib')

    # Predicting on the test data
    Predicted_Test = model.predict(X_test)

    y_test_list = y_test.iloc[:, 0].tolist()

    # Evaluating the model
    r2 = r2_score(y_test, Predicted_Test)
    mse = mean_squared_error(y_test, Predicted_Test)

    # Return the performance metrics
    return jsonify({
        "R2_Score": r2,
        "Mean_Squared_Error": mse,
        "actual": y_test_list,
        "predicted": Predicted_Test.tolist()
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)