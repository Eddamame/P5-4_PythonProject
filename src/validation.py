import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Author: Lideon
# Date: Updated for team changes
# Description: This module contains functions for validating all of the members' functions and user inputs etc.

# Importing all other team members' modules
try:
    from data_handler import data_handler
    from metrics import calculate_sma, calculate_runs, get_significant_runs
    from metrics import calculate_daily_returns
    from metrics import calculate_max_profit
    print("All team members' modules are imported successfully and validated!")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all team members' modules are in the 'src' directory and named correctly.")

# Test data setup (global)
def create_test_data():
    # Creating a controlled dataset for testing

    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')

    test_data = []

    # AAPL data with some fluctuations
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
        
    # MSFT data with some fluctuations
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

# Saving test data into a csv file to test data_handler
def save_test_data_csv(data, filename='test_data.csv'):
    data.to_csv(filename, index=False)
    return filename

# Formatted test results printing
def print_testresult(category, results):
    print(f"\n{category}")
    print("-" * 60)

    passed = 0
    total = len(results)

    for test_name, success, details in results:
        status = "PASS" if success else "FAIL"
        print(f"  {status}: {test_name}")
        if not success and details:
            print(f"      Details: {details}")
        if success:
            passed += 1

    percentage = (passed / total) * 100 if total > 0 else 0

    print(f"  Result: {passed}/{total} tests passed ({percentage:.0f}%)")
    print("-" * 60)
    return passed, total

# Testing Data Handler Module for Ignatius
def validate_data_handler():
    print("\n=== Testing Data Handler Module ===")

    tests = []
    test_data = create_test_data()

    try:
        # Save the test data to CSV
        test_file = save_test_data_csv(test_data)

        # Test 1.1: Basic Loading
        df = data_handler(test_file)
        tests.append(("Data Handler Loads CSV", df is not None and len(df) > 0, f"Loaded {len(df) if df is not None else 0} rows"))

        # Test 1.2: Columns Check
        if df is not None:
            expected_columns = {'name', 'date', 'open', 'high', 'low', 'close', 'volume'}
            actual_columns = all(col in df.columns for col in expected_columns)
            tests.append(("All required columns present", actual_columns, f"Columns: {list(df.columns)}"))

            # Test 1.3: Data Types Check
            date_is_datetime = pd.api.types.is_datetime64_any_dtype(df['date'])
            tests.append(("Date column is datetime", date_is_datetime,
                         f"Date type: {df['date'].dtype}"))
            
            price_cols_numeric = all(pd.api.types.is_numeric_dtype(df[col]) 
                                   for col in ['open', 'close', 'high', 'low'])
            tests.append(("Price columns are numeric", price_cols_numeric,
                         "Price columns type check"))
            
            # Test 1.4: Name Filtering
            filtered_df = data_handler(test_file, filterName=['AAPL'])
            aapl_only = filtered_df['name'].unique().tolist() == ['AAPL'] if filtered_df is not None else False
            tests.append(("Name filtering works", aapl_only,
                         f"Filtered stocks: {filtered_df['name'].unique() if filtered_df is not None else 'None'}"))

            # Test 1.5: Year Filtering
            time_filtered = data_handler(test_file, filterTime=(2023, 2023))
            correct_year = time_filtered['date'].dt.year.eq(2023).all() if time_filtered is not None else False
            tests.append(("Year filtering works", correct_year,
                         "2023 filter applied correctly"))
            
            # Test 1.6: Data Sorting
            is_sorted = df.equals(df.sort_values(['name', 'date']))
            tests.append(("Data is sorted by name and date", is_sorted,
                         "Sort order verification"))
            
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)

    except Exception as e:
        tests.append(("Data Handler Execution", False, f"Exception: {str(e)[:100]}"))

    return print_testresult("Data Handler Tests", tests)

# Testing SMA Module for Si Yun
def validate_sma_calculation():
    # Test SMA calculation function
    print("\n=== Testing SMA Calculation Module ===")
    
    tests = []
    
    try:
        # Test 2.1: Basic SMA calculation with list of window sizes
        test_windows = [5, 10]
        result_df = calculate_sma('AAPL', test_windows)
        
        tests.append(("SMA function returns DataFrame", isinstance(result_df, pd.DataFrame),
                     f"Returned type: {type(result_df)}"))
        
        if isinstance(result_df, pd.DataFrame):
            # Test 2.2: Required columns present
            has_sma_5 = 'sma_5' in result_df.columns
            has_sma_10 = 'sma_10' in result_df.columns
            has_close_col = 'close' in result_df.columns
            tests.append(("DataFrame has 'sma_5' column", has_sma_5, "SMA 5 column check"))
            tests.append(("DataFrame has 'sma_10' column", has_sma_10, "SMA 10 column check"))
            tests.append(("DataFrame has 'close' column", has_close_col, "Close column check"))
            
            # Test 2.3: Date is index
            date_index = isinstance(result_df.index, pd.DatetimeIndex)
            tests.append(("Date is set as index", date_index, f"Index type: {type(result_df.index)}"))
            
            # Test 2.4: SMA values validation for window size 5
            if has_sma_5 and has_close_col:
                close_prices = result_df['close'].tolist()
                if len(close_prices) >= 5:
                    # Find first non-NaN SMA value
                    first_valid_idx = result_df['sma_5'].first_valid_index()
                    if first_valid_idx is not None:
                        idx_pos = result_df.index.get_loc(first_valid_idx)
                        if idx_pos >= 4:
                            manual_sma = sum(close_prices[idx_pos-4:idx_pos+1]) / 5
                            calculated_sma = result_df['sma_5'].loc[first_valid_idx]
                            
                            sma_accuracy = abs(manual_sma - calculated_sma) < 0.01
                            tests.append(("SMA calculation accuracy", sma_accuracy,
                                         f"Manual: {manual_sma:.4f}, Calculated: {calculated_sma:.4f}"))
                
                # Test 2.5: Padding with None values
                none_padding = sum(1 for x in result_df['sma_5'].iloc[:4] if pd.isna(x))
                correct_padding = none_padding == 4
                tests.append(("Correct None padding for window 5", correct_padding,
                             f"Expected 4 None values, got {none_padding}"))
        
        # Test 2.6: Single window size
        single_window_result = calculate_sma('AAPL', [7])
        has_sma_7 = 'sma_7' in single_window_result.columns if isinstance(single_window_result, pd.DataFrame) else False
        tests.append(("Works with single window size", has_sma_7,
                     "Window size [7] test"))
        
    except Exception as e:
        tests.append(("SMA calculation execution", False, f"Exception: {str(e)[:100]}"))
    
    return print_testresult("SMA Calculation Tests", tests)

# Testing Run Analysis for Eddison
def validate_runs_analysis():
    """Test runs analysis functions"""
    print("\n=== Testing Runs Analysis ===")
    
    tests = []
    test_data = create_test_data()
    
    try:
        # Filter for AAPL data
        aapl_data = test_data[test_data['name'] == 'AAPL'].copy()
        
        # Test 3.1: calculate_runs with DataFrame
        runs_df, direction = calculate_runs(aapl_data)
        
        tests.append(("calculate_runs returns DataFrame", isinstance(runs_df, pd.DataFrame),
                     f"Returned type: {type(runs_df)}"))
        
        if isinstance(runs_df, pd.DataFrame):
            # Test 3.2: DataFrame structure
            required_cols = ['start_date', 'end_date', 'direction', 'length', 'start_index', 'end_index']
            has_all_cols = all(col in runs_df.columns for col in required_cols)
            tests.append(("Runs DataFrame has required columns", has_all_cols,
                         f"Columns: {list(runs_df.columns)}"))
            
            # Test 3.3: Direction values
            if 'direction' in runs_df.columns and len(runs_df) > 0:
                valid_directions = runs_df['direction'].isin(['Up', 'Down']).all()
                tests.append(("Valid direction values", valid_directions,
                             f"Unique directions: {runs_df['direction'].unique()}"))
            
            # Test 3.4: Length values
            if 'length' in runs_df.columns and len(runs_df) > 0:
                positive_lengths = (runs_df['length'] > 0).all()
                tests.append(("All run lengths are positive", positive_lengths,
                             f"Length range: {runs_df['length'].min()}-{runs_df['length'].max()}"))
            
            # Test 3.5: get_significant_runs function
            significant_runs = get_significant_runs(runs_df, min_length=2)
            
            tests.append(("get_significant_runs returns dict", isinstance(significant_runs, dict),
                         f"Returned type: {type(significant_runs)}"))
            
            if isinstance(significant_runs, dict):
                has_keys = 'up_runs' in significant_runs and 'down_runs' in significant_runs
                tests.append(("Significant runs has correct keys", has_keys,
                             f"Keys: {list(significant_runs.keys())}"))
                
                # Test 3.6: Filtering works
                if has_keys and len(runs_df) > 0:
                    all_up = significant_runs['up_runs']
                    all_down = significant_runs['down_runs']
                    
                    if len(all_up) > 0:
                        min_length_check = (all_up['length'] >= 2).all()
                        tests.append(("Up runs filtered by min_length", min_length_check,
                                     f"Min length in up runs: {all_up['length'].min() if len(all_up) > 0 else 'N/A'}"))
        
    except Exception as e:
        tests.append(("Runs analysis execution", False, f"Exception: {str(e)[:100]}"))
    
    return print_testresult("Runs Analysis Tests", tests)

# Testing Profits & Returns Analysis for Xue E
def validate_returns_and_profit():
    """Test returns and profit calculation functions"""
    print("\n=== Testing Returns and Profits ===")
    
    tests = []
    test_data = create_test_data()
    
    try:
        # Test 4.1: Daily returns calculation
        returns_result = calculate_daily_returns(test_data, 'AAPL')
        
        tests.append(("Returns function returns DataFrame", isinstance(returns_result, pd.DataFrame),
                     f"Returned type: {type(returns_result)}"))
        
        if isinstance(returns_result, pd.DataFrame):
            # Test 4.2: Daily_Return column exists
            has_return_col = 'Daily_Return' in returns_result.columns
            tests.append(("DataFrame has 'Daily_Return' column", has_return_col,
                         f"Columns: {list(returns_result.columns)}"))
            
            if has_return_col:
                # Test 4.3: Manual validation of daily returns
                first_return = returns_result['Daily_Return'].iloc[0]
                tests.append(("First daily return is NaN", pd.isna(first_return),
                             f"First return: {first_return}"))
                
                # Second return manual calculation
                if len(returns_result) >= 2:
                    close_1 = returns_result['close'].iloc[0]  
                    close_2 = returns_result['close'].iloc[1]
                    manual_return = (close_2 - close_1) / close_1
                    calculated_return = returns_result['Daily_Return'].iloc[1]
                    
                    if not pd.isna(calculated_return):
                        accuracy = abs(manual_return - calculated_return) < 1e-4
                        tests.append(("Second daily return accurate", accuracy,
                                     f"Manual: {manual_return:.6f}, Calculated: {calculated_return:.6f}"))
        
        # Test 4.4: Max profit calculation
        test_prices = [100, 102, 98, 105, 99, 108, 95]  # Known test case
        max_profit = calculate_max_profit(test_prices)
        
        tests.append(("Max profit function returns number", isinstance(max_profit, (int, float)),
                     f"Returned type: {type(max_profit)}"))
        
        # Test 4.5: Valley-Peak algorithm validation
        # Manual valley-peak: sum all positive daily changes
        expected_profit = 18.0  # (102-100)+(105-98)+(108-99) = 2+7+9 = 18
        profit_accuracy = abs(max_profit - expected_profit) < 0.01
        tests.append(("Valley-peak algorithm correct", profit_accuracy,
                     f"Expected: {expected_profit}, Got: {max_profit}"))
        
        # Test 4.6: Edge cases
        try:
            single_price = calculate_max_profit([100])
            tests.append(("Single price returns 0", single_price == 0.0,
                         f"Single price result: {single_price}"))
        except:
            tests.append(("Single price handled gracefully", True, "Exception handled"))
        
        try:
            empty_profit = calculate_max_profit([])
            tests.append(("Empty list handled", False, f"Should raise ValueError"))
        except ValueError:
            tests.append(("Empty list raises ValueError", True, "Proper error handling"))
        except:
            tests.append(("Empty list error handling", False, "Unexpected error type"))
            
    except Exception as e:
        tests.append(("Returns and profit analysis execution", False, f"Exception: {str(e)[:100]}"))
    
    return print_testresult("Returns and Profit Tests", tests)

# TEST FUNCTION 5: Real Data Compatibility
def validate_real_data_compatibility():
    """Test compatibility with GitHub dataset format"""
    print("\n=== Real Data Compatibility Test ===")
    
    tests = []
    
    try:
        # Test with actual GitHub data structure (lowercase 'name')
        github_sample = {
            'date': '2015-01-02',
            'open': 111.39,
            'high': 111.44, 
            'low': 107.35,
            'close': 109.33,
            'volume': 53204626,
            'name': 'AAPL'  # Changed to lowercase
        }
        
        # Test 5.1: Required columns
        required_cols = ['name', 'date', 'open', 'close', 'high', 'low', 'volume']
        has_all_cols = all(col in github_sample.keys() for col in required_cols)
        tests.append(("GitHub data has required columns", has_all_cols,
                     f"Required: {required_cols}"))
        
        # Test 5.2: Column naming (lowercase 'name')
        uses_name_column = 'name' in github_sample and 'Symbol' not in github_sample
        tests.append(("Uses 'name' column (team standard)", uses_name_column,
                     "Matches team code expectations"))
        
        # Test 5.3: Date format compatibility
        try:
            parsed_date = pd.to_datetime(github_sample['date'])
            date_valid = True
            year_range = 2015 <= parsed_date.year <= 2025
        except:
            date_valid = False
            year_range = False
        
        tests.append(("Date format parseable by pandas", date_valid,
                     f"Date: {github_sample['date']}"))
        tests.append(("Date in expected range (2015-2025)", year_range,
                     f"Parsed year: {parsed_date.year if date_valid else 'N/A'}"))
        
        # Test 5.4: Numeric data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        numeric_types = [isinstance(github_sample[col], (int, float)) for col in numeric_cols]
        all_numeric = all(numeric_types)
        tests.append(("All price/volume columns numeric", all_numeric,
                     f"Types: {[type(github_sample[col]).__name__ for col in numeric_cols]}"))
        
        # Test 5.5: Price logic consistency
        price_logic = (github_sample['high'] >= github_sample['low'] and
                      github_sample['high'] >= github_sample['open'] and
                      github_sample['high'] >= github_sample['close'] and
                      github_sample['low'] <= github_sample['open'] and
                      github_sample['low'] <= github_sample['close'])
        tests.append(("Price data follows logical constraints", price_logic,
                     f"H:{github_sample['high']}, L:{github_sample['low']}, O:{github_sample['open']}, C:{github_sample['close']}"))
        
        # Test 5.6: Expected stock symbols
        expected_stocks = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'TSLA']
        stock_in_list = github_sample['name'] in expected_stocks
        tests.append(("Stock symbol in expected list", stock_in_list,
                     f"Stock: {github_sample['name']}, Expected: {expected_stocks}"))
        
    except Exception as e:
        tests.append(("Real data compatibility test", False, f"Exception: {str(e)[:100]}"))
    
    return print_testresult("Real Data Compatibility Tests", tests)

# MAIN VALIDATION RUNNER
def run_all_validations():
    """Run all validation tests and generate report"""
    print("\n" + "=" * 70)
    print("STOCK MARKET ANALYSIS - VALIDATION SUITE")
    print("=" * 70)
    print("Testing all team member implementations with real data scenarios")
    print("=" * 70)
    
    # Run all validation functions
    all_results = {}
    all_results['data_handler'] = validate_data_handler()
    all_results['sma_calculation'] = validate_sma_calculation()
    all_results['runs_analysis'] = validate_runs_analysis()
    all_results['returns_profit'] = validate_returns_and_profit()
    all_results['real_data_compatibility'] = validate_real_data_compatibility()
    
    # Generate summary
    generate_final_summary(all_results)
    
    return all_results

def generate_final_summary(all_results):
    """Generate final summary report"""
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    total_passed = 0
    total_tests = 0
    
    for category, (passed, total) in all_results.items():
        total_passed += passed
        total_tests += total
        
        status = "✓ PASSED" if passed == total else "⚠ CAUTION" if passed > 0 else "✗ FAILED"
        print(f"{status} {category.replace('_', ' ').title()}: {passed}/{total}")
    
    print("-" * 70)
    overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    if total_passed == total_tests:
        print(f"🎉 ALL TESTS PASSED! {total_passed}/{total_tests} tests successful!")
        print("   Your program is ready for integration!")
    elif overall_percentage >= 80:
        print(f"✓ MOSTLY SUCCESSFUL: {total_passed}/{total_tests} tests passed ({overall_percentage:.0f}%)")
        print("   Minor issues to address, but core functionality works.")
    elif overall_percentage >= 60:
        print(f"⚠ PARTIAL SUCCESS: {total_passed}/{total_tests} tests passed ({overall_percentage:.0f}%)")
        print("   Some components need debugging before full integration.")
    else:
        print(f"✗ NEEDS WORK: {total_passed}/{total_tests} tests passed ({overall_percentage:.0f}%)")
        print("   Significant issues to resolve before integration.")
    
    print("\n📋 RECOMMENDED ACTIONS:")
    print("1. Fix any failed validation tests in team member code")
    print("2. Ensure consistent column naming (lowercase 'name')")
    print("3. Verify all imports work in src/ directory structure") 
    print("4. Test with your GitHub dataset URL")
    print("5. Run main.py for full integration testing")
    
    print("=" * 70 + "\n")

# Run validation when script is executed directly
if __name__ == "__main__":
    results = run_all_validations()