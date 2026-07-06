import urllib.request
import pandas as pd
import numpy as np
import io
import os

# Flight_info content was saved to this path by the previous tool step
flight_info_path = r"C:\Users\Lenova\.gemini\antigravity-ide\brain\8b334abd-c2d9-4650-a7f2-c925561fa7c8\.system_generated\steps\2450\content.md"

def load_flight_info():
    with open(flight_info_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Find where the CSV content starts (after '---')
    csv_part = content.split('---')[1].strip()
    df = pd.read_csv(io.StringIO(csv_part))
    return df

def download_flight_data(data_dir, flight_name):
    # Try the double directory format first: {data_dir}/{data_dir}/{flight_name}.csv
    url = f"https://raw.githubusercontent.com/YujiaoHu/AMOVFLY-Dataset/main/{data_dir}/{data_dir}/{flight_name}.csv"
    try:
        with urllib.request.urlopen(url) as response:
            csv_data = response.read().decode('utf-8')
        return pd.read_csv(io.StringIO(csv_data))
    except Exception as e1:
        # Try single directory format: {data_dir}/{flight_name}.csv
        url2 = f"https://raw.githubusercontent.com/YujiaoHu/AMOVFLY-Dataset/main/{data_dir}/{flight_name}.csv"
        try:
            with urllib.request.urlopen(url2) as response:
                csv_data = response.read().decode('utf-8')
            return pd.read_csv(io.StringIO(csv_data))
        except Exception as e2:
            print(f"Failed to download {flight_name} from {url} or {url2}: {e2}")
            return None

def main():
    print("Loading flight info...")
    info_df = load_flight_info()
    print(f"Total flights in metadata: {len(info_df)}")
    
    # Select a representative sample of flights from different dirs
    # We want to cover FAFS, FAVS, VAVS, Random to get a good distribution of speed and wind
    sampled_flights = []
    for dir_name in ['FAFS', 'FAVS', 'VAVS', 'Random']:
        sub = info_df[info_df['Data Dir'] == dir_name]
        if len(sub) > 0:
            # take up to 6 flights from each directory
            sampled_flights.extend(sub.head(6).to_dict('records'))
            
    print(f"Selected {len(sampled_flights)} flights for regression analysis.")
    
    all_v = []
    all_wind = []
    all_power = []
    
    for i, flight in enumerate(sampled_flights):
        data_dir = flight['Data Dir']
        flight_name = flight['FlightName']
        print(f"[{i+1}/{len(sampled_flights)}] Downloading {data_dir}/{flight_name}...")
        df = download_flight_data(data_dir, flight_name)
        if df is not None:
            # Check for required columns
            required = ['v_x', 'v_y', 'v_z', 'wind_speed', 'power']
            if all(col in df.columns for col in required):
                # Calculate ground speed
                v = np.sqrt(df['v_x']**2 + df['v_y']**2 + df['v_z']**2)
                wind = df['wind_speed']
                power = df['power']
                
                # Filter out points where power or speed is invalid (e.g. negative or extreme outliers)
                # Keep only valid flight points (exclude takeoff/landing where power is very low or speed is zero)
                valid = (power > 50) & (v >= 0) & (v <= 15) & (wind >= 0)
                all_v.extend(v[valid].tolist())
                all_wind.extend(wind[valid].tolist())
                all_power.extend(power[valid].tolist())
            else:
                print(f"Skipping {flight_name} due to missing columns. Columns present: {df.columns.tolist()}")

    v = np.array(all_v)
    wind = np.array(all_wind)
    power = np.array(all_power)
    
    print(f"\nTotal data points collected: {len(v)}")
    if len(v) == 0:
        print("No data collected. Exiting.")
        return
        
    # Fit OLS: power = beta_0 + beta_1 * v + beta_2 * v^2 + beta_3 * wind_speed
    # Design matrix X: [1, v, v^2, wind]
    X = np.column_stack((np.ones_like(v), v, v**2, wind))
    
    # Solve least squares: beta = (X^T X)^-1 X^T power
    beta, residuals, rank, s = np.linalg.lstsq(X, power, rcond=None)
    
    # Calculate R^2
    mean_power = np.mean(power)
    ss_tot = np.sum((power - mean_power)**2)
    ss_res = np.sum((power - X.dot(beta))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print("\nFitted Power Model in Watts:")
    print(f"P_meas = {beta[0]:.4f} + {beta[1]:.4f} * v + {beta[2]:.4f} * v^2 + {beta[3]:.4f} * w_wind")
    print(f"OLS R^2 = {r_squared:.4f}")
    
    # Convert to fractional capacity consumed per 10-second step
    # P_frac = P_meas * dt / (3600 * C_total)
    # Let's assume C_total = 60 Wh (typical capacity) and dt = 10s
    c_total = 60.0 # Wh
    dt = 10.0 # seconds
    scale = dt / (3600.0 * c_total)
    
    alpha = beta * scale
    
    print(f"\nFitted Fractional SoC Model per {dt}s step (assuming C_total = {c_total} Wh):")
    print(f"P_frac = {alpha[0]:.6f} + {alpha[1]:.6f} * v + {alpha[2]:.6f} * v^2 + {alpha[3]:.6f} * w_wind")
    print(f"Corresponding paper values: alpha_0=0.004, alpha_1=0.00015, alpha_2=0.00007, alpha_3=0.00025")
    
    # Let's find the C_total that would make alpha_0 closest to 0.004
    implied_c_total = (beta[0] * dt) / (3600.0 * 0.004)
    print(f"Implied C_total for alpha_0 to be exactly 0.004: {implied_c_total:.2f} Wh")
    
    # Let's calculate the fractional coefficients with this implied C_total
    scale_implied = dt / (3600.0 * implied_c_total)
    alpha_implied = beta * scale_implied
    print(f"\nFitted Fractional SoC Model with C_total = {implied_c_total:.2f} Wh:")
    print(f"alpha_0 (base hover)      = {alpha_implied[0]:.6f} (paper: 0.004000)")
    print(f"alpha_1 (linear speed)    = {alpha_implied[1]:.6f} (paper: 0.000150)")
    print(f"alpha_2 (quadratic speed) = {alpha_implied[2]:.6f} (paper: 0.000070)")
    print(f"alpha_3 (wind speed)      = {alpha_implied[3]:.6f} (paper: 0.000250)")

if __name__ == "__main__":
    main()
