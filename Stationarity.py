import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import yfinance as yf
from typing import List, Dict, Tuple




def check_stationarity(symbols: List[str], 
                       start_date: str = "2023-01-01", 
                       end_date: str = None,
                       data_source: str = "yfinance",
                       use_log_returns: bool = True) -> pd.DataFrame:
    
    results = []
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, 
                             show_errors=False)
            
            if data.empty:
                print(f"  ⚠ No data retrieved for {symbol}")
                continue
            
            if use_log_returns:
                series = np.log(data['Close'] / data['Close'].shift(1)).dropna()
                test_type = "Log Returns"
            else:
                series = data['Close'].dropna()
                test_type = "Price"
            
            adf_result = adfuller(series, autolag='AIC')
            adf_statistic = adf_result[0]
            adf_pvalue = adf_result[1]
            adf_is_stationary = adf_pvalue < 0.05
            
            kpss_result = kpss(series, regression='c', nlags='auto')
            kpss_statistic = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_is_stationary = kpss_pvalue > 0.05
            
            # Both tests must agree: ADF (p < 0.05) AND KPSS (p > 0.05)
            is_stationary = adf_is_stationary and kpss_is_stationary
            
            result = {
                'Symbol': symbol,
                'Test_Type': test_type,
                'Data_Points': len(series),
                'ADF_Statistic': adf_statistic,
                'ADF_P_Value': adf_pvalue,
                'ADF_Stationary': adf_is_stationary,
                'KPSS_Statistic': kpss_statistic,
                'KPSS_P_Value': kpss_pvalue,
                'KPSS_Stationary': kpss_is_stationary,
                'Overall_Stationary': is_stationary,
                'Mean': series.mean(),
                'Std_Dev': series.std(),
                'Status': '✓ Stationary' if is_stationary else '✗ Non-Stationary'
            }
            
            results.append(result)
            
            print(f"  {result['Status']}")
            print(f"    ADF p-value: {adf_pvalue:.6f} {'(Stationary)' if adf_is_stationary else '(Non-Stationary)'}")
            print(f"    KPSS p-value: {kpss_pvalue:.6f} {'(Stationary)' if kpss_is_stationary else '(Non-Stationary)'}")
            
        except Exception as e:
            print(f"  ✗ Error processing {symbol}: {str(e)}")
    
    return pd.DataFrame(results)




def check_stationarity_from_csv(csv_path: str, 
                                symbol_name: str,
                                price_column: str = 'Close',
                                use_log_returns: bool = True) -> Dict:
    """
    Check stationarity for a single symbol using data from a CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
    symbol_name : str
        Name/symbol for the data
    price_column : str
        Name of the price column (default: 'Close')
    use_log_returns : bool
        If True, tests log returns; if False, tests price series (default: True)
    
    Returns
    -------
    Dict
        Dictionary with stationarity test results
    """
    
    try:
        df = pd.read_csv(csv_path)
        
        if price_column not in df.columns:
            raise ValueError(f"Column '{price_column}' not found in CSV")
        
        if use_log_returns:
            series = np.log(df[price_column] / df[price_column].shift(1)).dropna()
        else:
            series = df[price_column].dropna()
        
        adf_result = adfuller(series, autolag='AIC')
        adf_is_stationary = adf_result[1] < 0.05
        
        kpss_result = kpss(series, regression='c', nlags='auto')
        kpss_is_stationary = kpss_result[1] > 0.05
        
        is_stationary = adf_is_stationary and kpss_is_stationary
        
        result = {
            'Symbol': symbol_name,
            'ADF_Statistic': adf_result[0],
            'ADF_P_Value': adf_result[1],
            'ADF_Stationary': adf_is_stationary,
            'KPSS_Statistic': kpss_result[0],
            'KPSS_P_Value': kpss_result[1],
            'KPSS_Stationary': kpss_is_stationary,
            'Overall_Stationary': is_stationary,
            'Status': '✓ Stationary' if is_stationary else '✗ Non-Stationary'
        }
        
        print(f"{symbol_name}: {result['Status']}")
        return result
        
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return None


if __name__ == "__main__":
    symbols = ["EURUSD=X", "AAPL", "MSFT"]
    results_df = check_stationarity(symbols, start_date="2023-01-01")
    
    print("\n" + "="*80)
    print("STATIONARITY TEST RESULTS")
    print("="*80)
    print(results_df.to_string())
    
    results_df.to_csv('stationarity_results.csv', index=False)
    print("\nResults saved to 'stationarity_results.csv'")
