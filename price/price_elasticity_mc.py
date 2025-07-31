import numpy as np
import pandas as pd
import json

# Open the JSON file and load its contents
with open('/Users/ryanborman/Documents/GitHub/last_mile/datasets/data.json', 'r') as file:
    data = json.load(file)
print(data)

# 1. Define the problem
remaining_deliveries = 20
prices_to_test = [10, 15, 20, 25]  # Test a range of prices
num_simulations = 10000

# 2. Assume a price elasticity of demand.
# A value of 2 means a 10% price increase leads to a 20% rise in deliveries taken.
# This would be calculated from historical data in a real scenario.
elasticity = 2 # Elasticity = (% Change in Quantity) / (% Change in Price)
base_price = 10
base_demand = 20  # The target number of deliveries

def simulate_demand(price, base_price, base_demand, elasticity, noise_level=2):
    """Simulates the number of deliveries taken at a given price."""
    # Calculate expected demand based on price elasticity
    price_change_percent = (price - base_price) / base_price
    demand_change_percent = elasticity * price_change_percent
    expected_demand = base_demand * (1 + demand_change_percent)

    # Add randomness to simulate real-world variability
    simulated_demand = np.random.normal(expected_demand, noise_level)

    # Ensure demand is not negative and does not exceed the total available
    return max(0, min(simulated_demand, base_demand))


# 4. Run the Monte Carlo simulation
simulation_results = {}
for price in prices_to_test:
    deliveries_completed = [simulate_demand(price, base_price, base_demand, elasticity) for _ in range(num_simulations)]

    # Analyze the results
    avg_deliveries = np.mean(deliveries_completed)
    prob_all_completed = np.mean([d >= remaining_deliveries for d in deliveries_completed])
    total_cost = price * avg_deliveries

    simulation_results[price] = {
        'avg_deliveries_taken': avg_deliveries,
        'probability_all_completed': prob_all_completed,
        'expected_total_cost': total_cost
    }

# 5. Find the optimal price
# The optimal price is the lowest price with a high probability of completing all deliveries.
optimal_price = None
for price, results in simulation_results.items():
    if results['probability_all_completed'] > 0.95:  # Target a 95% success rate
        if optimal_price is None or price < optimal_price:
            optimal_price = price

# Print results
results_df = pd.DataFrame(simulation_results).T
print("Simulation Results:")
print(results_df)
print(f"\\nOptimal price to ensure all deliveries are taken: ${optimal_price}")

