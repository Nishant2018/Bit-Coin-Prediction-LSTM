# Bitcoin Prediction Flask Web Application

## Introduction to Bitcoin
Bitcoin is a decentralized digital currency that operates without a central authority or single administrator. It was invented in 2008 by an unknown person or group of people using the name Satoshi Nakamoto and was released as open-source software in 2009. Bitcoin transactions are verified by network nodes through cryptography and recorded in a public distributed ledger called a blockchain. Bitcoin is unique in that there are a finite number of them: 21 million.

![Bitcoin](https://github.com/Nishant2018/Bit-Coin-Prediction-LSTM/blob/main/static/css/1.jpg)

Bitcoin's value has seen significant fluctuations over time, making it a popular asset for traders and investors. The purpose of this web application is to predict future Bitcoin prices using historical data and advanced machine learning models.
    
## How to Run This Flask Web App       
        
### Prerequisites
Before running the Flask web application, make sure you have the following installed on your system:
- Python 3.x
- pip (Python package installer)
- Flask (`pip install flask`)
- Other necessary libraries (like TensorFlow, Pandas, etc.)

### Steps to Run the Application
1. **Clone the Repository**: First, clone the repository to your local machine using the following command:
    ```bash
    git clone https://github.com/your-username/bitcoin-prediction-app.git
    ```
    
2. **Navigate to the Project Directory**: Move into the directory where the project files are located:
    ```bash
    cd bitcoin-prediction-app
    ```

3. **Install Dependencies**: Install all required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Flask Application**: Start the Flask web server by executing:
    ```bash
    python app.py
    ```

5. **Access the Web App**: Open your web browser and go to:
    ```
    http://127.0.0.1:5000
    ```
    You should see the Bitcoin prediction web app running.

### Customizing the Application
- **Change the Bitcoin Dataset**: You can update the dataset used for predictions by replacing the existing CSV file in the `data` directory.
- **Modify Prediction Model**: If you wish to use a different prediction model (e.g., LSTM, RNN), modify the model code in the `model.py` file.

```markdown

