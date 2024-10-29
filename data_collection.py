import requests
import pandas as pd
import numpy as np
import datetime as dt
import time

API_KEY = ''

def get_historical_data(ticker, start_date, end_date):
    """
    Fetch historical daily price and volume data for a given ticker.
    """
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}'
    params = {
        'apiKey': API_KEY,
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000
    }
    response = requests.get(url, params=params)
    data = response.json()
    if 'results' in data:
        df = pd.DataFrame(data['results'])
        df['date'] = pd.to_datetime(df['t'], unit='ms')
        df = df[['date', 'o', 'h', 'l', 'c', 'v']]
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        return df
    else:
        return pd.DataFrame()

# data_collection.py
def get_all_tickers():
    """
    Fetch all US stock tickers from Polygon.io and save to a CSV file.
    """
    base_url = 'https://api.polygon.io/v3/reference/tickers'
    params = {
        'apiKey': API_KEY,
        'market': 'stocks',
        'active': 'true',
        'locale': 'us',
        'limit': 1000  # Max limit per request
    }

    tickers = []
    url = base_url
    page = 1  # To keep track of the page number

    while True:
        print(f"\nFetching page {page}...")
        print(f"Request URL: {url}")
        print(f"Parameters: {params}")

        try:
            response = requests.get(url, params=params)
            print(f"HTTP Status Code: {response.status_code}")
            
            # Check if the request was successful
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                print(f"Response: {response.text}")
                break

            data = response.json()
            print(f"Response Count: {data.get('count', 'N/A')}")
            print(f"Number of Results in this page: {len(data.get('results', []))}")

            # Extract tickers from the current page
            if 'results' in data:
                for item in data['results']:
                    tickers.append(item['ticker'])
                print(f"Total tickers collected so far: {len(tickers)}")
            else:
                print("No 'results' key found in the response.")
                break

            # Handle pagination using 'next_url'
            next_url = data.get('next_url', None)
            if next_url:
                print(f"Next URL found: {next_url}")
                url = next_url  # Set the URL to the next page
                params = {}  # Parameters are already included in 'next_url'
                page += 1
                time.sleep(0.2)  # Pause to respect API rate limits
            else:
                print("No further pages. Pagination complete.")
                break

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break
        except ValueError as ve:
            print(f"JSON decoding failed: {ve}")
            break
        except Exception as ex:
            print(f"An unexpected error occurred: {ex}")
            break

    # Save all collected tickers to a CSV file
    if tickers:
        df = pd.DataFrame(tickers, columns=['ticker'])
        df.to_csv('tickers.csv', index=False)
        print(f"\nSaved {len(tickers)} tickers to 'tickers.csv'")
    else:
        print("\nNo tickers were collected.")

# data_collection.py

from urllib.parse import urlparse, parse_qs, urlencode, urlunparse



def get_all_tickers():
    """
    Fetch all US stock tickers from Polygon.io and save to a CSV file.
    """
    base_url = 'https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=1000&apiKey='
    params = {
        'apiKey': API_KEY,
        'market': 'stocks',
        'active': 'true',
        'locale': 'us',
        'limit': 1000  # Max limit per request
    }

    tickers = []
    url = base_url
    page = 1  # To keep track of the page number

    while True:
        # Add apiKey to the first request
        current_url = url  # next_url already includes apiKey

        print(f"\nFetching page {page}...")
        print(f"Request URL: {current_url}")
        print(f"Parameters: {params if page == 1 else 'N/A (Using next_url)'}")

        try:
            response = requests.get(current_url)
            print(f"HTTP Status Code: {response.status_code}")

            # Check if the request was successful
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                print(f"Response: {response.text}")
                break

            data = response.json()
            print(f"Response Count: {data.get('count', 'N/A')}")
            print(f"Number of Results in this page: {len(data.get('results', []))}")

            # Extract tickers from the current page
            if 'results' in data:
                for item in data['results']:
                    tickers.append(item['ticker'])
                print(f"Total tickers collected so far: {len(tickers)}")
            else:
                print("No 'results' key found in the response.")
                break

            # Handle pagination using 'next_url'
            next_url = data.get('next_url', None)
            if next_url:
                print(f"Next URL found: {next_url}")
                # Ensure the next_url includes the apiKey
                next_url_with_api = next_url + f'&apiKey={API_KEY}'
                url = next_url_with_api  # Set the URL to the next page with apiKey
                page += 1
                time.sleep(0.2)  # Be polite with API rate limits
            else:
                print("No further pages. Pagination complete.")
                break

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break
        except ValueError as ve:
            print(f"JSON decoding failed: {ve}")
            break
        except Exception as ex:
            print(f"An unexpected error occurred: {ex}")
            break

    # Save all collected tickers to a CSV file
    if tickers:
        df = pd.DataFrame(tickers, columns=['ticker'])
        df.to_csv('tickers.csv', index=False)
        print(f"\nSaved {len(tickers)} tickers to 'tickers.csv'")
    else:
        print("\nNo tickers were collected.")

