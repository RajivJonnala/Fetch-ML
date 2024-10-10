# Receipt Prediction App

This is a Flask web application that predicts monthly receipt counts for 2022 using a trained logarithmic regression model. The app takes user input for the desired year and returns predictions along with visualizations of historical and predicted data.

## Features

- **User Input**: Users can input the year for which they want to predict receipt counts.
- **Data Visualization**: The app displays a plot of historical receipt counts alongside the predicted values for the specified year.
- **Model Training**: A logarithmic regression model is trained on monthly receipt data.

## Requirements

To run this application, you'll need the following:

- Python 3.x
- Flask
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

You can install the necessary packages using pip:

```bash
pip install flask torch torchvision torchaudio pandas numpy matplotlib scikit-learn
```
## Project Structure
    ```bash
    /receipt_prediction_app
    ├── app.py                # Main Flask application
    ├── data_daily.csv       # CSV file containing daily receipt data
    ├── logarithmic_model.pth # Trained model weights
    ├── LogarithmicRegression.py #  Model
    └── templates
        └── index.html       # HTML template for user interface
    ```
## How to Run the Application
1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/RajivJonnala/Fetch-ML.git
    ```
2. Make sure you have the required data file (data_daily.csv) and trained model (logarithmic_model.pth) in the project directory.
3. Run the application:
    ```bash
    python app.py
    ```
4. Open your web browser and navigate to http://127.0.0.1:5000/ to access the app.

## Usage
The app will display a plot showing the historical receipt counts and the predicted values for the 2021-2022.
#
