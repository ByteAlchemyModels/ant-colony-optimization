import pandas as pd
import numpy as np
import statsmodels.api as sm

def calculate_price_elasticity(data, city, method='log-log'):
    """
    Calculates the price elasticity of demand for a city using one of two methods.

    The function can compute elasticity using either a log-log regression model,
    which provides a single constant elasticity estimate, or by averaging the
    period-over-period percentage changes (arc elasticity).

    Args:
        data (pd.DataFrame): A DataFrame with columns 'date', 'city',
                             'average_price', and 'volume'.
        city (str): The name of the city for which to calculate elasticity.
        method (str, optional): The calculation method. Accepts:
                                - 'log-log' (default): Uses OLS regression on log-transformed data.
                                - 'percentage_change': Averages the point-to-point arc elasticities.
                                Defaults to 'log-log'.

    Returns:
        float: The calculated price elasticity of demand for the specified city.
               Returns NaN if the calculation cannot be completed.

    Raises:
        ValueError: If an unsupported method is specified.
    """
    # Filter the dataset for the specified city
    city_data = data[data['city'] == city].copy()

    # --- Method 1: Log-Log Regression (Constant Elasticity) ---
    if method == 'log-log':
        # Ensure price and volume are positive for log transformation
        city_data = city_data[(city_data['average_price'] > 0) & (city_data['volume'] > 0)]

        if len(city_data) < 2:
            return float('nan')  # Not enough data to run regression

        # Apply the natural logarithm to price and volume
        city_data['log_price'] = np.log(city_data['average_price'])
        city_data['log_volume'] = np.log(city_data['volume'])

        # Define the independent (X) and dependent (y) variables
        X = city_data['log_price']
        y = city_data['log_volume']

        # Add a constant (intercept) to the model
        X = sm.add_constant(X)

        # Fit the Ordinary Least Squares (OLS) model
        model = sm.OLS(y, X).fit()

        # The coefficient of log_price is the elasticity
        elasticity = model.params.get('log_price', float('nan'))
        return elasticity

    # --- Method 2: Averaged Percentage Change (Arc Elasticity) ---
    elif method == 'percentage_change':
        # Convert 'date' to datetime and sort chronologically
        city_data['date'] = pd.to_datetime(city_data['date'])
        city_data = city_data.sort_values('date')

        # Calculate the percentage change in price and volume
        city_data['price_pct_change'] = city_data['average_price'].pct_change()
        city_data['volume_pct_change'] = city_data['volume'].pct_change()

        # Drop the first row which will have NaN values
        city_data = city_data.dropna(subset=['price_pct_change', 'volume_pct_change'])

        # Calculate elasticity, handling division by zero (where price is unchanged)
        # Replacing inf with NaN allows us to ignore these points in the mean calculation
        city_data['elasticity'] = city_data['volume_pct_change'] / city_data['price_pct_change']
        city_data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Return the average of the point-to-point elasticities
        return city_data['elasticity'].mean()

    # --- Handle Invalid Method ---
    else:
        raise ValueError("Invalid method specified. Please choose 'log-log' or 'percentage_change'.")


# --- Example Usage ---

# Load the dataset from the provided file
file_path = '/Users/ryanborman/Documents/GitHub/last_mile/datasets/datasetsdelivery_data.csv'
delivery_data = pd.read_csv(file_path)

# Calculate elasticity for New York using both methods
citys_to_analyze = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
for city in citys_to_analyze:
    # 1. Using the default 'log-log' method
    elasticity_log = calculate_price_elasticity(delivery_data, city, method='log-log')
    print(f"Price Elasticity for {city} (Log-Log Method): {elasticity_log:.4f}")

    # 2. Using the 'percentage_change' method
    elasticity_pct = calculate_price_elasticity(delivery_data, city, method='percentage_change')
    print(f"Price Elasticity for {city} (Percentage Change Method): {elasticity_pct:.4f}")
