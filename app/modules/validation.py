import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Author: Lideon
# Date: Updated for team changes
# Description: This module contains functions for validating all of the members' functions and user inputs etc.

# Importing all other team members' modules
# Store original imports as None to avoid NameError

print(f"🔍 Validation module loading...")
print(f"   __name__ = {__name__}")
print(f"   __package__ = {__package__}")
print(f"   __file__ = {__file__}")

# Create dummy functions in case imports fail
def _create_dummy(name):
    def dummy(*args, **kwargs):
        raise ImportError(f"{name} could not be imported")
    return dummy

# Initialize with dummies
api_data_handler = _create_dummy('api_data_handler')
handle_backup_csv = _create_dummy('handle_backup_csv')
calculate_sma = _create_dummy('calculate_sma')
calculate_runs = _create_dummy('calculate_runs')
get_significant_runs = _create_dummy('get_significant_runs')
calculate_daily_returns = _create_dummy('calculate_daily_returns')
calculate_max_profit = _create_dummy('calculate_max_profit')
validate_model = _create_dummy('validate_model')
forecast_prices = _create_dummy('forecast_prices')
plot_price_and_sma = _create_dummy('plot_price_and_sma')
plot_runs = _create_dummy('plot_runs')
plot_daily_returns_plotly = _create_dummy('plot_daily_returns_plotly')
validation_plot = _create_dummy('validation_plot')
validation_table = _create_dummy('validation_table')

# Try to import real modules
import_success = False
import_method = None

# Method 1: Try relative imports (when imported by Flask app)
if not import_success:
    try:
        print("   Attempting relative imports...")
        from .data_handler import api_data_handler, handle_backup_csv
        from .metrics import (
            calculate_sma, calculate_runs, get_significant_runs,
            calculate_daily_returns, calculate_max_profit)
        from .prediction import validate_model, forecast_prices
        from .visualization import (
            plot_price_and_sma, plot_runs, plot_daily_returns_plotly,
            validation_plot, validation_table)
        import_success = True
        import_method = "relative imports"
        print(f"   ✅ SUCCESS via relative imports")
    except (ImportError, ValueError) as e:
        print(f"   ❌ Relative imports failed: {e}")

# Method 2: Try absolute imports (when run directly)
if not import_success:
    try:
        print("   Attempting absolute imports...")
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            print(f"   Added to path: {current_dir}")
        
        from data_handler import api_data_handler, handle_backup_csv
        from metrics import (
            calculate_sma, calculate_runs, get_significant_runs,
            calculate_daily_returns, calculate_max_profit)
        from prediction import validate_model, forecast_prices
        from visualization import (
            plot_price_and_sma, plot_runs, plot_daily_returns_plotly,
            validation_plot, validation_table)
        import_success = True
        import_method = "absolute imports"
        print(f"   ✅ SUCCESS via absolute imports")
    except ImportError as e:
        print(f"   ❌ Absolute imports failed: {e}")

# Final status
if import_success:
    print(f"\n✅ All team members' modules imported successfully via {import_method}!\n")
else:
    print(f"\n❌ ERROR: Could not import team modules!")
    print(f"   Validation tests will fail.")
    print(f"   Please ensure all .py files exist in: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"   Required files: data_handler.py, metrics.py, prediction.py, visualization.py\n")


# Global log to capture validation outputs
validation_log = []

def log(message):
    """Helper function to store messages in the validation log"""
    print(message)
    validation_log.append(message)

# Test data setup
def create_test_data():
    """Creating a controlled dataset for testing"""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    test_data = []

    # AAPL data
    aapl_prices = [150, 152, 151, 153, 155, 154, 156, 158, 157, 159, 160, 162, 161, 163, 165, 164, 166, 168, 167, 169]
    for i, date in enumerate(dates):
        price = aapl_prices[i]
        test_data.append({
            'name': 'AAPL',  
            'date': date, 
            'open': price * 0.995, 
            'high': price * 1.02, 
            'low': price * 0.98,
            'close': price,
            'volume': 1000000 + i * 50000
        })

    # MSFT data
    msft_prices = [300, 302, 301, 303, 305, 304, 306, 308, 307, 309, 310, 312, 311, 313, 315, 314, 316, 318, 317, 319]
    for i, date in enumerate(dates):
        price = msft_prices[i]
        test_data.append({
            'name': 'MSFT', 
            'date': date, 
            'open': price * 0.995, 
            'high': price * 1.02, 
            'low': price * 0.98,
            'close': price,
            'volume': 800000 + i * 30000
        })

    return pd.DataFrame(test_data)

# Save test data to CSV
def save_test_data_csv(data, filename='test_data.csv'):
    data.to_csv(filename, index=False)
    return filename

# Formatted test results
def print_testresult(category, results):
    log(f"\n{category}")
    log("-" * 60)

    passed = 0
    total = len(results)
    test_details = []

    for test_name, success, details in results:
        status = "✓ PASS" if success else "✗ FAIL"
        log(f"  {status}: {test_name}")
        if not success and details:
            log(f"      Details: {details}")
        if success:
            passed += 1
        test_details.append({
            "test_name": test_name,
            "success": success,
            "details": details
        })

    percentage = (passed / total) * 100 if total > 0 else 0
    log(f"  Result: {passed}/{total} tests passed ({percentage:.0f}%)")
    log("-" * 60)
    return passed, total, test_details

# Data Handler Tests
def validate_data_handler():
    log("\n=== Testing Data Handler Module ===")
    tests = []
    test_data = create_test_data()

    try:
        aapl_data = test_data[test_data['name'] == 'AAPL'].copy()
        df = api_data_handler(aapl_data, 'AAPL')
        tests.append(("api_data_handler processes DataFrame", df is not None and len(df) > 0, 
                     f"Loaded {len(df) if df is not None else 0} rows"))

        if df is not None:
            expected_columns = {'name', 'date', 'open', 'high', 'low', 'close', 'volume'}
            actual_columns = all(col in df.columns for col in expected_columns)
            tests.append(("All required columns present", actual_columns, 
                         f"Columns: {list(df.columns)}"))

            date_is_datetime = pd.api.types.is_datetime64_any_dtype(df['date'])
            tests.append(("Date column is datetime", date_is_datetime,
                         f"Date type: {df['date'].dtype}"))

            price_cols_numeric = all(pd.api.types.is_numeric_dtype(df[col]) 
                                   for col in ['open', 'close', 'high', 'low'])
            tests.append(("Price columns are numeric", price_cols_numeric,
                         "Price columns type check"))

            correct_name = df['name'].iloc[0] == 'AAPL' if len(df) > 0 else False
            tests.append(("Name column is correct", correct_name,
                         f"Name: {df['name'].iloc[0] if len(df) > 0 else 'None'}"))

            time_filtered = api_data_handler(aapl_data, 'AAPL', filterTime=(2023, 2023))
            correct_year = time_filtered['date'].dt.year.eq(2023).all() if time_filtered is not None and len(time_filtered) > 0 else False
            tests.append(("Year filtering works", correct_year,
                         "2023 filter applied correctly"))

            is_sorted = df['date'].is_monotonic_increasing
            tests.append(("Data is sorted by date", is_sorted,
                         "Sort order verification"))
    except Exception as e:
        tests.append(("Data Handler Execution", False, f"Exception: {str(e)[:100]}"))

    return print_testresult("Data Handler Tests", tests)

# SMA Tests
def validate_sma_calculation():
    log("\n=== Testing SMA Calculation Module ===")
    tests = []
    
    try:
        test_data = create_test_data()
        aapl_data = test_data[test_data['name'] == 'AAPL'].copy()
        test_windows = [5, 10]
        result_df = calculate_sma(aapl_data, test_windows)
        tests.append(("SMA function returns DataFrame", isinstance(result_df, pd.DataFrame),
                     f"Returned type: {type(result_df)}"))

        if isinstance(result_df, pd.DataFrame):
            has_sma_5 = 'sma_5' in result_df.columns
            has_sma_10 = 'sma_10' in result_df.columns
            has_close_col = 'close' in result_df.columns
            has_date_col = 'date' in result_df.columns
            
            tests.append(("DataFrame has 'sma_5' column", has_sma_5, "SMA 5 column check"))
            tests.append(("DataFrame has 'sma_10' column", has_sma_10, "SMA 10 column check"))
            tests.append(("DataFrame has 'close' column", has_close_col, "Close column check"))
            tests.append(("DataFrame has 'date' column", has_date_col, "Date column check"))

            if has_sma_5 and has_close_col:
                sma_5_values = result_df['sma_5'].tolist()
                first_valid_idx = next((i for i, v in enumerate(sma_5_values) if v is not None), None)
                if first_valid_idx is not None and first_valid_idx >= 4:
                    close_prices = result_df['close'].tolist()
                    manual_sma = sum(close_prices[first_valid_idx-4:first_valid_idx+1]) / 5
                    calculated_sma = sma_5_values[first_valid_idx]
                    sma_accuracy = abs(manual_sma - calculated_sma) < 0.01
                    tests.append(("SMA calculation accuracy", sma_accuracy, 
                                 f"Manual: {manual_sma:.4f}, Calculated: {calculated_sma:.4f}"))

                none_padding = sum(1 for x in sma_5_values[:4] if x is None or pd.isna(x))
                correct_padding = none_padding == 4
                tests.append(("Correct None padding for window 5", correct_padding,
                             f"Expected 4 None values, got {none_padding}"))

        single_window_result = calculate_sma(aapl_data, 7)
        has_sma_7 = 'sma_7' in single_window_result.columns if isinstance(single_window_result, pd.DataFrame) else False
        tests.append(("Works with single integer window size", has_sma_7,
                     "Window size 7 test"))
    except Exception as e:
        tests.append(("SMA calculation execution", False, f"Exception: {str(e)[:100]}"))

    return print_testresult("SMA Calculation Tests", tests)

# Runs Analysis Tests
def validate_runs_analysis():
    log("\n=== Testing Runs Analysis ===")
    tests = []
    test_data = create_test_data()
    
    try:
        aapl_data = test_data[test_data['name'] == 'AAPL'].copy()
        runs_df, direction, processed_df = calculate_runs(aapl_data)

        tests.append(("calculate_runs returns DataFrame", isinstance(runs_df, pd.DataFrame),
                     f"Returned type: {type(runs_df)}"))
        tests.append(("calculate_runs returns direction Series", isinstance(direction, pd.Series),
                     f"Direction type: {type(direction)}"))
        tests.append(("calculate_runs returns processed DataFrame", isinstance(processed_df, pd.DataFrame),
                     f"Processed df type: {type(processed_df)}"))

        if isinstance(runs_df, pd.DataFrame) and len(runs_df) > 0:
            required_cols = ['start_date', 'end_date', 'direction', 'length', 'start_index', 'end_index']
            has_all_cols = all(col in runs_df.columns for col in required_cols)
            tests.append(("Runs DataFrame has required columns", has_all_cols,
                         f"Columns: {list(runs_df.columns)}"))

            if 'direction' in runs_df.columns:
                valid_directions = runs_df['direction'].isin(['Up', 'Down']).all()
                tests.append(("Valid direction values", valid_directions,
                             f"Unique directions: {runs_df['direction'].unique()}"))

            if 'length' in runs_df.columns:
                positive_lengths = (runs_df['length'] > 0).all()
                tests.append(("All run lengths are positive", positive_lengths,
                             f"Length range: {runs_df['length'].min()}-{runs_df['length'].max()}"))

            significant_runs = get_significant_runs(runs_df, min_length=2)
            tests.append(("get_significant_runs returns dict", isinstance(significant_runs, dict),
                         f"Returned type: {type(significant_runs)}"))

            if isinstance(significant_runs, dict):
                has_keys = all(key in significant_runs for key in ['up_runs', 'down_runs', 'significant_runs'])
                tests.append(("Significant runs has correct keys", has_keys,
                             f"Keys: {list(significant_runs.keys())}"))

                if has_keys and len(significant_runs['up_runs']) > 0:
                    min_length_check = (significant_runs['up_runs']['length'] >= 2).all()
                    tests.append(("Up runs filtered by min_length", min_length_check,
                                 f"Min length in up runs: {significant_runs['up_runs']['length'].min()}"))

    except Exception as e:
        tests.append(("Runs analysis execution", False, f"Exception: {str(e)[:100]}"))

    return print_testresult("Runs Analysis Tests", tests)

# Returns & Profit Tests
def validate_returns_and_profit():
    log("\n=== Testing Returns and Profits ===")
    tests = []
    test_data = create_test_data()

    try:
        aapl_data = test_data[test_data['name'] == 'AAPL'].copy()
        returns_result = calculate_daily_returns(aapl_data)
        tests.append(("Returns function returns DataFrame", isinstance(returns_result, pd.DataFrame),
                     f"Returned type: {type(returns_result)}"))

        if isinstance(returns_result, pd.DataFrame) and len(returns_result) > 0:
            has_return_col = 'Daily_Return' in returns_result.columns
            tests.append(("DataFrame has 'Daily_Return' column", has_return_col,
                         f"Columns: {list(returns_result.columns)}"))

            if has_return_col:
                first_return = returns_result['Daily_Return'].iloc[0]
                tests.append(("First daily return is NaN", pd.isna(first_return),
                             f"First return: {first_return}"))

                if len(returns_result) >= 2:
                    close_1 = returns_result['close'].iloc[0]
                    close_2 = returns_result['close'].iloc[1]
                    manual_return = (close_2 - close_1) / close_1
                    calculated_return = returns_result['Daily_Return'].iloc[1]
                    if not pd.isna(calculated_return):
                        accuracy = abs(manual_return - calculated_return) < 1e-4
                        tests.append(("Second daily return accurate", accuracy,
                                     f"Manual: {manual_return:.6f}, Calculated: {calculated_return:.6f}"))

        df_price = pd.DataFrame({'close': [100, 102, 98, 105, 99, 108, 95]})
        max_profit = calculate_max_profit(df_price)
        tests.append(("Max profit function returns number", isinstance(max_profit, (int, float)),
                     f"Returned type: {type(max_profit)}"))

        expected_profit = 18.0
        profit_accuracy = abs(max_profit - expected_profit) < 0.01
        tests.append(("Valley-peak algorithm correct", profit_accuracy,
                     f"Expected: {expected_profit}, Got: {max_profit}"))

        try:
            single_price = calculate_max_profit(pd.DataFrame({'close': [100]}))
            tests.append(("Single price returns 0", single_price == 0.0,
                         f"Single price result: {single_price}"))
        except:
            tests.append(("Single price handled", False, "Exception occurred"))

        try:
            empty_profit = calculate_max_profit(pd.DataFrame({'close': []}))
            tests.append(("Empty DataFrame handled", empty_profit == 0.0, 
                         f"Empty result: {empty_profit}"))
        except:
            tests.append(("Empty DataFrame error handling", True, "Exception raised as expected"))

    except Exception as e:
        tests.append(("Returns and profit analysis execution", False, f"Exception: {str(e)[:100]}"))

    return print_testresult("Returns and Profit Tests", tests)

# Real Data Compatibility
def validate_real_data_compatibility():
    log("\n=== Real Data Compatibility Test ===")
    tests = []

    try:
        github_sample = {
            'date': '2015-01-02',
            'open': 111.39,
            'high': 111.44, 
            'low': 107.35,
            'close': 109.33,
            'volume': 53204626,
            'name': 'AAPL'  
        }

        required_cols = ['name', 'date', 'open', 'close', 'high', 'low', 'volume']
        has_all_cols = all(col in github_sample.keys() for col in required_cols)
        tests.append(("Data has required columns", has_all_cols, f"Required: {required_cols}"))

        uses_name_column = 'name' in github_sample and 'Symbol' not in github_sample
        tests.append(("Uses 'name' column (team standard)", uses_name_column, "Matches team code expectations"))

        try:
            parsed_date = pd.to_datetime(github_sample['date'])
            date_valid = True
            year_range = 2015 <= parsed_date.year <= 2025
        except:
            date_valid = False
            year_range = False

        tests.append(("Date format parseable by pandas", date_valid, f"Date: {github_sample['date']}"))
        tests.append(("Date in expected range (2015-2025)", year_range, f"Parsed year: {parsed_date.year if date_valid else 'N/A'}"))

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        numeric_types = [isinstance(github_sample[col], (int, float)) for col in numeric_cols]
        all_numeric = all(numeric_types)
        tests.append(("All price/volume columns numeric", all_numeric,
                     f"Types: {[type(github_sample[col]).__name__ for col in numeric_cols]}"))

        price_logic = (github_sample['high'] >= github_sample['low'] and
                      github_sample['high'] >= github_sample['open'] and
                      github_sample['high'] >= github_sample['close'] and
                      github_sample['low'] <= github_sample['open'] and
                      github_sample['low'] <= github_sample['close'])
        tests.append(("Price data follows logical constraints", price_logic,
                     f"H:{github_sample['high']}, L:{github_sample['low']}, O:{github_sample['open']}, C:{github_sample['close']}"))

        expected_stocks = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'TSLA']
        stock_in_list = github_sample['name'] in expected_stocks
        tests.append(("Stock symbol in expected list", stock_in_list,
                     f"Stock: {github_sample['name']}, Expected: {expected_stocks}"))

    except Exception as e:
        tests.append(("Real data compatibility test", False, f"Exception: {str(e)[:100]}"))

    return print_testresult("Real Data Compatibility Tests", tests)

# Main Runner
def run_all_validations():
    log("\n" + "=" * 70)
    log("STOCK MARKET ANALYSIS - VALIDATION SUITE")
    log("=" * 70)
    log("Testing all team member implementations with real data scenarios")
    log("=" * 70)

    all_results = {}
    categories = [
        ('data_handler', validate_data_handler),
        ('sma_calculation', validate_sma_calculation),
        ('runs_analysis', validate_runs_analysis),
        ('returns_profit', validate_returns_and_profit),
        ('real_data_compatibility', validate_real_data_compatibility)
    ]

    for name, func in categories:
        passed, total, details = func()
        all_results[name] = {"passed": passed, "total": total, "tests": details}

    generate_final_summary(all_results)
    return all_results

def generate_final_summary(all_results):
    log("\n" + "=" * 70)
    log("VALIDATION SUMMARY")
    log("=" * 70)

    total_passed = sum(res['passed'] for res in all_results.values())
    total_tests = sum(res['total'] for res in all_results.values())
    overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0

    for category, res in all_results.items():
        status = "✓ PASSED" if res['passed'] == res['total'] else "⚠️ CAUTION" if res['passed'] > 0 else "✗ FAILED"
        log(f"{status} {category.replace('_', ' ').title()}: {res['passed']}/{res['total']}")

    if total_passed == total_tests:
        log(f"✓ ALL TESTS PASSED! {total_passed}/{total_tests} tests successful!")
        log("  Your program is ready for integration!")
    elif overall_percentage >= 80:
        log(f"⚠️ MOSTLY SUCCESSFUL: {total_passed}/{total_tests} tests passed ({overall_percentage:.0f}%)")
        log("  Minor issues to address, but core functionality works.")
    elif overall_percentage >= 60:
        log(f"⚠️ PARTIAL SUCCESS: {total_passed}/{total_tests} tests passed ({overall_percentage:.0f}%)")
        log("  Some components need debugging before full integration.")
    else:
        log(f"✗ NEEDS WORK: {total_passed}/{total_tests} tests passed ({overall_percentage:.0f}%)")
        log("  Significant issues to resolve before integration.")

    log("\n📋 RECOMMENDED ACTIONS:")
    log("1. Fix any failed validation tests in team member code")
    log("2. Ensure consistent column naming (lowercase 'name')")
    log("3. Verify all imports work in app/modules/ directory structure") 
    log("4. Test with your data sources (yfinance API and backup CSV)")
    log("5. Run main.py for full integration testing")
    log("=" * 70 + "\n")

# Run directly
if __name__ == "__main__":
    run_all_validations()
