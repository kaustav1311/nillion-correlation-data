#!/usr/bin/env python3
"""
Script to fetch and store historical price data for cryptocurrency correlation analysis.
This script fetches data for BTC, ETH, NIL and MIND Network for correlation analysis.
"""

import os
import json
import time
import requests
import numpy as np
from datetime import datetime

# Create output directory if it doesn't exist
os.makedirs('data/correlation', exist_ok=True)

# CoinGecko API base URL
COINGECKO_API = "https://api.coingecko.com/api/v3"

# List of tokens to track
TOKENS = {
    'bitcoin': 'BTC',
    'ethereum': 'ETH',
    'nillion': 'NIL',
    'mind-network': 'MIND'
}

# Time frames to collect data for
TIMEFRAMES = [
    {'name': '30d', 'days': 30},
    {'name': '90d', 'days': 90}
]

def fetch_price_data(coin_id, days):
    """
    Fetch historical price data for a cryptocurrency from CoinGecko.
    
    Args:
        coin_id: CoinGecko API coin ID
        days: Number of days of historical data to fetch
        
    Returns:
        List of price data points
    """
    endpoint = f"{COINGECKO_API}/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    
    # Add substantial delay to avoid API restrictions
    print(f"  Waiting 30 seconds before fetching {coin_id} data...")
    time.sleep(30)  # Sleep for 10 seconds
    
    try:
        print(f"  Sending request for {coin_id}...")
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Format the price data
        price_data = []
        for timestamp, price in data['prices']:
            # Convert milliseconds to seconds timestamp
            price_data.append({
                'timestamp': timestamp,
                'price': price
            })
        
        print(f"  Successfully fetched {len(price_data)} data points for {coin_id}")
        return price_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {coin_id} data: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error with {coin_id}: {e}")
        return []

def calculate_correlation(prices1, prices2):
    """
    Calculate Pearson correlation coefficient between two price series.
    
    Args:
        prices1: First price series
        prices2: Second price series
        
    Returns:
        Correlation coefficient
    """
    if len(prices1) != len(prices2) or len(prices1) < 2:
        return 0
    
    # Calculate daily returns
    returns1 = []
    returns2 = []
    
    for i in range(1, len(prices1)):
        r1 = (prices1[i] - prices1[i-1]) / prices1[i-1]
        r2 = (prices2[i] - prices2[i-1]) / prices2[i-1]
        returns1.append(r1)
        returns2.append(r2)
    
    # Calculate correlation
    return np.corrcoef(returns1, returns2)[0, 1]

def save_data_for_timeframe(timeframe):
    """
    Fetch, process and save data for a specific timeframe.
    
    Args:
        timeframe: Dictionary containing timeframe information
    """
    timeframe_name = timeframe['name']
    days = timeframe['days']
    
    print(f"Fetching data for {timeframe_name} timeframe...")
    
    # Fetch data for all tokens
    token_data = {}
    failed_tokens = []
    
    for coin_id in TOKENS.keys():
        print(f"  Fetching {coin_id} data...")
        price_data = fetch_price_data(coin_id, days)
        if price_data:
            # Save raw data for each token
            token_filename = f"data/correlation/{coin_id}_{timeframe_name}.json"
            with open(token_filename, 'w') as f:
                json.dump(price_data, f, indent=2)
                
            # Store data for correlation calculation
            token_data[coin_id] = price_data
        else:
            failed_tokens.append(coin_id)
            print(f"  WARNING: Failed to fetch data for {coin_id}")
    
    # Check if we have enough data to proceed
    if len(token_data) < 2:
        print(f"  ERROR: Not enough token data available for {timeframe_name}. Skipping correlation calculation.")
        return
    
    if 'nillion' not in token_data:
        print(f"  ERROR: Nillion data is missing for {timeframe_name}. Skipping correlation calculation.")
        return
    
    print(f"  Processing data for {timeframe_name}...")
    
    # Align timestamps for correlation calculation
    timestamps = set()
    for data in token_data.values():
        for point in data:
            # Round to nearest day to handle slight timestamp differences
            timestamp = point['timestamp'] // 86400000 * 86400000
            timestamps.add(timestamp)
    
    timestamps = sorted(timestamps)
    
    # Create price maps for each token
    price_maps = {}
    for coin_id, data in token_data.items():
        price_map = {}
        for point in data:
            timestamp = point['timestamp'] // 86400000 * 86400000
            price_map[timestamp] = point['price']
        price_maps[coin_id] = price_map
    
    # Align data and calculate correlations
    aligned_data = {
        'dates': [],
        'prices': {}
    }
    
    for coin_id in token_data.keys():
        aligned_data['prices'][coin_id] = []
    
    # Find common timestamps where we have data for available tokens
    common_timestamps = []
    for ts in timestamps:
        if all(ts in price_maps.get(coin_id, {}) for coin_id in token_data.keys()):
            common_timestamps.append(ts)
    
    if not common_timestamps:
        print(f"  ERROR: No common timestamps found for {timeframe_name}. Skipping correlation calculation.")
        return
    
    # Create aligned data arrays
    for ts in common_timestamps:
        date = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d')
        aligned_data['dates'].append(date)
        
        for coin_id in token_data.keys():
            aligned_data['prices'][coin_id].append(price_maps[coin_id][ts])
    
    # Save aligned data
    aligned_filename = f"data/correlation/aligned_{timeframe_name}.json"
    with open(aligned_filename, 'w') as f:
        json.dump(aligned_data, f, indent=2)
    
    # Calculate correlations
    if 'nillion' in aligned_data['prices'] and len(aligned_data['dates']) > 1:
        nil_prices = aligned_data['prices']['nillion']
        
        correlations = {}
        for coin_id in token_data.keys():
            if coin_id != 'nillion':
                token_prices = aligned_data['prices'][coin_id]
                correlation = calculate_correlation(nil_prices, token_prices)
                token_symbol = TOKENS[coin_id].lower()
                correlations[f'nil_{token_symbol}'] = correlation
                
        # Save correlation data
        correlation_filename = f"data/correlation/correlation_{timeframe_name}.json"
        with open(correlation_filename, 'w') as f:
            json.dump(correlations, f, indent=2)
        
        print(f"  Correlations for {timeframe_name}:")
        for pair, value in correlations.items():
            print(f"    {pair}: {value:.4f}")

def main():
    """Main execution function."""
    for timeframe in TIMEFRAMES:
        save_data_for_timeframe(timeframe)
    
    print("Data collection complete!")

if __name__ == "__main__":
    main()
