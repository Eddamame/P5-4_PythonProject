# INF1002 Stock Market Analysis Project

## Overview

The **Stock Market Analysis Project** is a web application built with **Python Flask** that allows users to perform historical and predictive analysis on any given stock ticker. It is now deployed on **Railway.app**. Users can input a ticker symbol and a time period to instantly generate a dashboard with various technical analysis visualizations, including Moving Averages, Daily Returns, Market Price Runs, Maximum Profit calculation, and a next-day price prediction.

## Features

* **Historical Data Fetching:** Utilizes the `yfinance` API to fetch accurate historical stock data.
* **Multiple Timeframes:** Supports analysis periods of up to 6 months, 1 year, 2 years, or 3 years.
* **Next-Day Price Prediction:** Features a dedicated predictive model to forecast the next day's closing price.
* **Technical Analysis Tools:**
    * Simple Moving Average (SMA) with 20, 50, and 200-day window options.
    * Daily Percentage Returns.
    * Market Price Runs (identifying consecutive up/down days).
    * Maximum Profit calculation over the analysis period.
* **Interactive Visualizations:** Generates clear and insightful charts for all selected metrics.
* **Automated Validation Suite:** Built-in testing framework to validate all modules and ensure data integrity.

---

## Project Structure 📁

The application now uses the **Application Factory pattern** within the `app/` package, with `main.py` serving as the new entry point, and utilizes a `Procfile` for **Railway.app** deployment.
P5-4_PythonProject/
├── app/
│   ├── modules/                     # Core Business & ML Logic
|   |   ├── init.py
│   │   ├── data_fetcher.py
│   │   ├── data_handler.py
│   │   ├── metrics.py
│   │   ├── model.py                 # Dedicated file for the core predictive model
│   │   ├── prediction.py
|   |   ├── validation.py            # Automated testing suite for all modules and functions
│   │   └── visualization.py
│   ├── static/                      # Static Assets (CSS and Images)
│   │   ├── css/
│   │   │   └── style.css
│   │   └── img/
│   │       └── logo.png
│   ├── templates/                   # HTML Templates
│   │   ├── index.html
│   │   ├── metrics.html
│   │   ├── results.html
|   |   └── validation.html
│   ├── init.py                      # Flask App Factory (e.g., create_app())
│   └── routes.py                    # Flask Route Definitions (Blueprint)
├── data/                            # Contingency CSV files or model artifacts
│   └── backup.csv
├── main.py                          # Entry point for local and production
├── .gitignore                       # for Git/Railway: Specifies files to ignore
├── Procfile                         # for Railway: Defines the start command
├── requirements.txt                 # for Railway: Lists all Python dependencies
└── README.md


| File | Role | Key Functions |
| :--- | :--- | :--- |
| **`main.py`** | **Application Entry Point.** Initializes and runs the Flask app from the factory. | `app = create_app()`, `app.run()` |
| **`app/__init__.py`** | **Flask App Factory.** Creates the application instance and registers the `routes` blueprint. | `def create_app()` |
| **`app/routes.py`** | Core application logic and URL routing blueprint. | `@bp.route('/')`, `@bp.route('/metrics')` |
| `app/modules/data_fetcher.py` | Fetches raw data from the `yfinance` API. | `get_hist_data(ticker, period)` |
| `app/modules/data_handler.py` | Cleans and transforms the raw fetched data. | `api_data_handler(data)` |
| `app/modules/model.py` | Contains the structure and loading of the machine learning model. | `load_model()` |
| `app/modules/prediction.py` | Handles the next-day price prediction logic. | `predict_next_day(data)` |
| `app/modules/metrics.py` | Calculates all technical metrics. | `calculate_daily_returns(data)`, `calculate_sma(data)` |
| `app/modules/validation.py` | Automated testing suite for all modules and functions. | `run_all_validations()`, `validate_data_handler()`, `validate_sma_calculation()`, `validate_runs_analysis()`, `validate_returns_and_profit()` |
| `app/modules/visualization.py` | Generates all necessary charts and plots. | `plot_sma()`, `plot_runs()` |
| **`Procfile`** | **Railway Deployment Command.** | `web: gunicorn main:app` (Example) |

---

## Installation and Setup 

### Prerequisites

You need **Python 3.8+** installed on your system.

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Eddamame/P5-4_PythonProject]
    cd P5-4_PythonProject
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/Mac
    venv\Scripts\activate     # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application (using the new entry point):**
    ```bash
    python main.py
    ```
    The application should now be accessible at `http://127.0.0.1:8080/`.

---

## Deployment on Railway.app 🚀

This project is configured for seamless deployment on **Railway.app** using the `Procfile` to define the start command.

1.  **Procfile:** Ensure your `Procfile` is correctly configured to run the WSGI server (like Gunicorn):
    ```
    web: gunicorn main:app
    ```
    *(This assumes `main.py` is configured to expose the application instance as the variable `app`)*

2.  **Connect & Deploy:** Connect your GitHub repository to a new project on Railway. Railway will automatically detect the Python environment, install dependencies from `requirements.txt`, and use the `Procfile` command to start the web service.

---

## Usage Flow

The application follows a three-step routing process:

### Step I: Input (`index.html` -> `/`)

* **Action:** User inputs a stock **Ticker** and selects an analysis **Time Period**.
* **Logic:** `app/routes.py` initiates data fetching via `data_fetcher.py` and cleaning via `data_handler.py`.

### Step II: Metric Selection (`metrics.html` -> `/metrics`)

* **Action:** User selects analysis methodologies (SMA, Daily Returns, Market Price Runs, Max Profit) they want to view.
* **Logic:** The cleaned data is held, and `app/modules/metrics.py` functions are prepared for execution based on selections.

### Step III: Results Dashboard (`results.html` -> `/results`)

* **Action:** The user is presented with the final dashboard.
* **Logic:**
    * `metrics.py` calculates the selected metrics.
    * `prediction.py` runs the forecast using the model logic.
    * `visualization.py` generates charts using the calculated results.

## Validation & Testing

The project includes a comprehensive validation system (validation.py) that tests all core modules to ensure data integrity and function correctness.

### Running Validation Tests
* **Option 1: Web Interface**
    1. Start the Flask application: python main.py
    2. Navigate to http://127.0.0.1:5000/validate
    3. View real-time test results in the browser dashboard

* **Option 2: Command Line**
    #### Run validation directly
    python app/modules/validation.py

    #### Or via Python module
    python -m app.modules.validation

## What Gets Validated
* **The validation suite runs 38+ automated tests across 5 categories:**

|`Test Category`	        `Tests`	`What It Validates`|
* **Data Handler**	            7	API data processing, column presence, data types, date parsing, filtering sorting

**SMA Calculation**	            7	Moving average accuracy, window sizes, padding, DataFrame structure

**Runs Analysis**	            9	Run detection, direction classification, length calculation, filtering

**Returns & Profit**	        8	Daily return calculation, max profit algorithm, edge cases

**Real Data Compatibility**	    7	Column naming, data formats, logical constraints, stock symbol validation   

---

## Contributing

We welcome contributions! If you have suggestions or would like to improve the codebase, please feel free to:

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License 📄

This project is licensed under the **MIT License**.

The MIT License is a simple, permissive open-source license. It grants users the freedom to:

* **Use** the software for any purpose.
* **Modify** the software.
* **Distribute** the software.
* **Sublicense** the software.
* **Sell** copies of the software.

The only condition is that the license and copyright notice must be included in all copies or substantial portions of the software.
For the full license text, see the [`LICENSE`](LICENSE) file in the repository.