# Use this for your visualization functions 
import matplotlib.pyplot as plt
from src.sma import calculate_sma  

def plot_price_and_sma(stock_name, window_size):
    # Calculate SMA 
    filtered_df = calculate_sma(stock_name, window_size)
    # give a dimensions to a width of 10 inches and a height of 6 inches
    plt.figure(figsize=(10, 6))
      
    # Plot close price
    plt.plot(filtered_df.index, filtered_df['close'],label='Close Price', color='blue')
    
    # Plot SMA
    plt.plot(filtered_df.index, filtered_df['sma'], label=f'SMA {window_size}', color='green')
    
    # Label the graph and provide title
    plt.title(f"Stock Price with SMA{window_size} for {stock_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
