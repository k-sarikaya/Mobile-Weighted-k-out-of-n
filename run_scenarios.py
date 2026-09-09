import os
import math
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import norm, t as t_dist

class Agent:
    def __init__(self, idx, atype, weight, rs, lambda_rate):
        self.idx = idx
        self.atype = atype
        self.weight = weight
        self.rs = rs
        self.lambda_rate = lambda_rate
        self.x = 0.0
        self.y = 0.0
        self.active = True
        self.soc = 1.0  # State of Charge (100% battery)

def create_swarm():
    swarm = []
    idx = 0
    # Type A
    for _ in range(3):
        swarm.append(Agent(idx, 'A', 3, 150, 0.005))
        idx += 1
    # Type B
    for _ in range(5):
        swarm.append(Agent(idx, 'B', 2, 100, 0.008))
        idx += 1
    # Type C
    for _ in range(4):
        swarm.append(Agent(idx, 'C', 1, 50, 0.012))
        idx += 1
    return swarm

def check_mission_success_proposed_and_baselines(swarm, Rc, K_threshold, H_max, L_max=100.0):
    active_agents = [a for a in swarm if a.active]
    n_active = len(active_agents)
    
    # 1. Baseline 1: Classic Static Weighted k-out-of-n:G (Eryilmaz 2013)
    total_active_weight = sum(a.weight for a in active_agents)
    B1_t = 1 if total_active_weight >= K_threshold else 0
    
    # 2. Baseline 3: Consecutive-k-out-of-n:F Swarm (Dui 2021)
    B3_t = 1
    for idx in range(len(swarm) - 2):
        if not swarm[idx].active and not swarm[idx+1].active and not swarm[idx+2].active:
            B3_t = 0
            break
            
    if n_active == 0:
        return 0, 0, 0, B1_t, 0, B3_t

    # Probabilistic connection probability (FANET sigmoidal fit)
    adj = {a.idx: [] for a in active_agents}
    for i in range(n_active):
        for j in range(i+1, n_active):
            a1 = active_agents[i]
            a2 = active_agents[j]
            d_ij = math.hypot(a1.x - a2.x, a1.y - a2.y)
            p_conn = 1.0 / (1.0 + math.exp(0.05 * (d_ij - 250.0)))
            if np.random.rand() <= p_conn:
                adj[a1.idx].append(a2.idx)
                adj[a2.idx].append(a1.idx)
                
    visited = set()
    components = []
    for a in active_agents:
        if a.idx not in visited:
            comp = []
            queue = [a.idx]
            visited.add(a.idx)
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                comp.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            components.append(comp)
            
    if not components:
        return 0, 0, 0, B1_t, 0, B3_t
        
    largest_cc = max(components, key=len)
    weight_map = {a.idx: a.weight for a in active_agents}
    cc_weight = sum(weight_map[node] for node in largest_cc)
    
    # 3. Baseline 2: Dynamic Multistate Network without Weights (Xu 2022)
    B2_t = 1 if len(largest_cc) >= 9 else 0
    
    total_active_weight = sum(a.weight for a in active_agents)
    A_t = 1 if total_active_weight >= K_threshold else 0
    B_t = 1 if cc_weight >= K_threshold else 0
    
    # Latency constraint using Log-Normal hop delays
    C_t = 0
    if B_t == 1:
        nodes_set = set(largest_cc)
        hops_list = []
        for start_node in largest_cc:
            dist = {start_node: 0}
            queue = [start_node]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if v in nodes_set and v not in dist:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for target_node, hops in dist.items():
                if hops > 0:
                    hops_list.append(hops)
                    
        if hops_list:
            total_hops = sum(hops_list)
            # Vectorised batch sampling of all path delays at once
            all_lats = np.random.lognormal(mean=2.0, sigma=0.5, size=total_hops)
            idx = 0
            max_latency = 0.0
            for hops in hops_list:
                lat = all_lats[idx : idx + hops].sum()
                idx += hops
                if lat > max_latency:
                    max_latency = lat
            if max_latency <= L_max:
                C_t = 1
        else:
            C_t = 1
            
    M_t = 1 if (B_t == 1 and C_t == 1) else 0
    return A_t, B_t, M_t, B1_t, B2_t, B3_t

def generate_copula_samples(n, copula_type, rho, df=3):
    if copula_type == 'indep':
        return np.random.uniform(0, 1, n)
    elif copula_type == 'gaussian':
        cov = np.full((n, n), rho)
        np.fill_diagonal(cov, 1.0)
        Z = np.random.multivariate_normal(np.zeros(n), cov)
        return norm.cdf(Z)
    elif copula_type == 't':
        cov = np.full((n, n), rho)
        np.fill_diagonal(cov, 1.0)
        chi2 = np.random.chisquare(df)
        Z = np.random.multivariate_normal(np.zeros(n), cov)
        X = Z * np.sqrt(df / chi2)
        return t_dist.cdf(X, df)
    elif copula_type == 'clayton':
        theta = rho
        if theta <= 0.0:
            return np.random.uniform(0, 1, n)
        # Gamma mixture representation for Clayton copula
        V = np.random.gamma(1.0 / theta, 1.0)
        V = max(V, 1e-12)
        X = np.random.exponential(1.0, n)
        return (1.0 + X / V) ** (-1.0 / theta)
    elif copula_type == 'gumbel':
        theta = rho
        if theta <= 1.0:
            return np.random.uniform(0, 1, n)
        alpha = 1.0 / theta
        # Chambers-Mallows-Stuck algorithm for positive stable variable V
        U_val = np.random.uniform(-math.pi / 2.0, math.pi / 2.0)
        W = np.random.exponential(1.0)
        term1 = math.sin(alpha * (U_val + math.pi / 2.0)) / (math.cos(U_val) ** (1.0 / alpha))
        term2 = (math.cos(U_val - alpha * (U_val + math.pi / 2.0)) / W) ** ((1.0 - alpha) / alpha)
        V = max(term1 * term2, 1e-12)
        X = np.random.exponential(1.0, n)
        return np.exp(-(X / V) ** alpha)

def simulate_run(scenario, max_steps=120):
    swarm = create_swarm()
    Rc = 250.0
    K_threshold = 18
    H_max = 4
    L_max = 100.0  # Max latency bound of 100ms
    
    mode = scenario['mode']
    copula = scenario.get('copula', 'indep')
    rho = scenario.get('rho', 0.5)
    df = scenario.get('df', 3)
    
    mission_status = []
    n_agents = len(swarm)
    
    # Initialize agent-specific systematic flight variations for stochastic mobility
    for i, a in enumerate(swarm):
        a.speed_mult = np.random.uniform(0.9, 1.1)
        a.angle_offset = np.random.uniform(-0.1, 0.1)
        a.soc = 1.0  # Reset State of Charge
        
    # Pre-generate Time-to-Failure (TTF) using Copula to align with continuous degradation theory
    if mode in ['S2', 'S3']:
        U = generate_copula_samples(n_agents, copula, rho, df)
        for i, a in enumerate(swarm):
            a.tau = -math.log(1.0 - U[i]) / a.lambda_rate
    else:
        for a in swarm:
            a.tau = float('inf')
            
    for step in range(max_steps):
        # Position update and energy depletion rate (calibrated with AMOVFLY dataset)
        if mode in ['S1', 'S3']:
            # Expand formation: spiral out
            wind_factor = 1.0
            wind_speed = 0.0
            if mode == 'S3':
                # wind adds extra random drift per step
                wind_factor = np.random.uniform(0.9, 1.1)
                wind_speed = np.random.uniform(5.0, 15.0)  # Dynamic wind speed in m/s
                
            for i, a in enumerate(swarm):
                expansion_factor = (20.0 + step * 2.5) * a.speed_mult * wind_factor
                angle = i * (2 * math.pi / n_agents) + a.angle_offset
                a.x = expansion_factor * math.cos(angle)
                a.y = expansion_factor * math.sin(angle)
                
                # Speed proportional to spatial expansion rate
                v_i = 2.5 * a.speed_mult
                # Empirical battery draw formula from AMOVFLY telemetry
                power_draw = (0.004 + 0.00015 * v_i + 0.00007 * (v_i**2) + 0.00025 * wind_speed) * a.speed_mult
                a.soc -= power_draw
        else:
            # S2: tight cluster
            for i, a in enumerate(swarm):
                angle = i * (2 * math.pi / n_agents)
                a.x = 40.0 * math.cos(angle)
                a.y = 40.0 * math.sin(angle)
                
                # Low hover speed, zero wind
                v_i = 0.5 * a.speed_mult
                power_draw = (0.003 + 0.0001 * v_i) * a.speed_mult
                a.soc -= power_draw
                
        # Health update based on pre-calculated TTF and energy depletion threshold (10%)
        for a in swarm:
            if step >= a.tau or a.soc <= 0.10:
                a.active = False
            
        A_t, B_t, M_t, B1_t, B2_t, B3_t = check_mission_success_proposed_and_baselines(swarm, Rc, K_threshold, H_max, L_max)
        mission_status.append({
            'step': step,
            'A_t': A_t,
            'B_t': B_t,
            'M_t': M_t,
            'B1_t': B1_t,
            'B2_t': B2_t,
            'B3_t': B3_t
        })
        
    return mission_status

def worker_experiment_run(arg):
    scenario, run_id = arg
    np.random.seed(42 + run_id)  # Ensure unique, reproducible random seed for each child process
    res = simulate_run(scenario)
    
    # Calculate survival curves for this run
    survived_so_far = True
    b1_survived = True
    b2_survived = True
    b3_survived = True
    
    curve = []
    b1_curve = []
    b2_curve = []
    b3_curve = []
    
    for step_data in res:
        if step_data['M_t'] == 0:
            survived_so_far = False
        if step_data['B1_t'] == 0:
            b1_survived = False
        if step_data['B2_t'] == 0:
            b2_survived = False
        if step_data['B3_t'] == 0:
            b3_survived = False
            
        curve.append(1 if survived_so_far else 0)
        b1_curve.append(1 if b1_survived else 0)
        b2_curve.append(1 if b2_survived else 0)
        b3_curve.append(1 if b3_survived else 0)
        
    return res, curve, b1_curve, b2_curve, b3_curve

def run_experiment(scenario, n_runs=5000):
    from multiprocessing import Pool
    
    # Parallelize the runs of the experiment
    args = [(scenario, r) for r in range(n_runs)]
    with Pool() as pool:
        results = pool.map(worker_experiment_run, args)
        
    all_runs = []
    survival_curves = []
    b1_survival_curves = []
    b2_survival_curves = []
    b3_survival_curves = []
    
    for r, (res, curve, b1_curve, b2_curve, b3_curve) in enumerate(results):
        survival_curves.append(curve)
        b1_survival_curves.append(b1_curve)
        b2_survival_curves.append(b2_curve)
        b3_survival_curves.append(b3_curve)
        
        df = pd.DataFrame(res)
        df['run'] = r
        all_runs.append(df)
        
    combined = pd.concat(all_runs)
    avg_per_step = combined.groupby('step').mean().reset_index()
    
    survival_curves = np.array(survival_curves)
    b1_survival_curves = np.array(b1_survival_curves)
    b2_survival_curves = np.array(b2_survival_curves)
    b3_survival_curves = np.array(b3_survival_curves)
    
    R_mis = np.mean(survival_curves, axis=0)
    R_mis_se = np.sqrt(R_mis * (1.0 - R_mis) / n_runs)
    
    avg_per_step['R_mis'] = R_mis
    avg_per_step['R_mis_se'] = R_mis_se
    avg_per_step['R_b1'] = np.mean(b1_survival_curves, axis=0)
    avg_per_step['R_b1_se'] = np.sqrt(avg_per_step['R_b1'] * (1.0 - avg_per_step['R_b1']) / n_runs)
    avg_per_step['R_b2'] = np.mean(b2_survival_curves, axis=0)
    avg_per_step['R_b2_se'] = np.sqrt(avg_per_step['R_b2'] * (1.0 - avg_per_step['R_b2']) / n_runs)
    avg_per_step['R_b3'] = np.mean(b3_survival_curves, axis=0)
    avg_per_step['R_b3_se'] = np.sqrt(avg_per_step['R_b3'] * (1.0 - avg_per_step['R_b3']) / n_runs)
    
    avg_per_step['A_t_se'] = np.sqrt(avg_per_step['A_t'] * (1.0 - avg_per_step['A_t']) / n_runs)
    avg_per_step['B_t_se'] = np.sqrt(avg_per_step['B_t'] * (1.0 - avg_per_step['B_t']) / n_runs)
    
    return avg_per_step

def simulate_run_fast(scenario, max_steps=120):
    swarm = create_swarm()
    K_threshold = 18
    L_max = 100.0
    
    mode = scenario['mode']
    copula = scenario.get('copula', 'indep')
    rho = scenario.get('rho', 0.5)
    df = scenario.get('df', 3)
    
    n_agents = len(swarm)
    
    for i, a in enumerate(swarm):
        a.speed_mult = np.random.uniform(0.9, 1.1)
        a.angle_offset = np.random.uniform(-0.1, 0.1)
        a.soc = 1.0
        
    if mode in ['S2', 'S3']:
        U = generate_copula_samples(n_agents, copula, rho, df)
        for i, a in enumerate(swarm):
            a.tau = -math.log(1.0 - U[i]) / a.lambda_rate
    else:
        for a in swarm:
            a.tau = float('inf')
            
    ttf = max_steps
    for step in range(max_steps):
        if mode in ['S1', 'S3']:
            wind_factor = 1.0
            wind_speed = 0.0
            if mode == 'S3':
                wind_factor = np.random.uniform(0.9, 1.1)
                wind_speed = np.random.uniform(5.0, 15.0)
                
            for i, a in enumerate(swarm):
                expansion_factor = (20.0 + step * 2.5) * a.speed_mult * wind_factor
                angle = i * (2 * math.pi / n_agents) + a.angle_offset
                a.x = expansion_factor * math.cos(angle)
                a.y = expansion_factor * math.sin(angle)
                
                v_i = 2.5 * a.speed_mult
                power_draw = (0.004 + 0.00015 * v_i + 0.00007 * (v_i**2) + 0.00025 * wind_speed) * a.speed_mult
                a.soc -= power_draw
        else:
            for i, a in enumerate(swarm):
                angle = i * (2 * math.pi / n_agents)
                a.x = 40.0 * math.cos(angle)
                a.y = 40.0 * math.sin(angle)
                
                v_i = 0.5 * a.speed_mult
                power_draw = (0.003 + 0.0001 * v_i) * a.speed_mult
                a.soc -= power_draw
                
        for a in swarm:
            if step >= a.tau or a.soc <= 0.10:
                a.active = False
            
        # Fast proposed model failure check
        active_agents = [a for a in swarm if a.active]
        n_active = len(active_agents)
        if n_active == 0:
            ttf = step
            break
            
        total_active_weight = sum(a.weight for a in active_agents)
        if total_active_weight < K_threshold:
            ttf = step
            break
            
        adj = {a.idx: [] for a in active_agents}
        for i in range(n_active):
            for j in range(i+1, n_active):
                a1 = active_agents[i]
                a2 = active_agents[j]
                d_ij = math.hypot(a1.x - a2.x, a1.y - a2.y)
                p_conn = 1.0 / (1.0 + math.exp(0.05 * (d_ij - 250.0)))
                if np.random.rand() <= p_conn:
                    adj[a1.idx].append(a2.idx)
                    adj[a2.idx].append(a1.idx)
                    
        visited = set()
        components = []
        for a in active_agents:
            if a.idx not in visited:
                comp = []
                queue = [a.idx]
                visited.add(a.idx)
                head = 0
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    comp.append(u)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            queue.append(v)
                components.append(comp)
                
        if not components:
            ttf = step
            break
            
        largest_cc = max(components, key=len)
        weight_map = {a.idx: a.weight for a in active_agents}
        cc_weight = sum(weight_map[node] for node in largest_cc)
        
        if cc_weight < K_threshold:
            ttf = step
            break
            
        C_t = 0
        nodes_set = set(largest_cc)
        hops_list = []
        for start_node in largest_cc:
            dist = {start_node: 0}
            queue = [start_node]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if v in nodes_set and v not in dist:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for target_node, hops in dist.items():
                if hops > 0:
                    hops_list.append(hops)
                    
        if hops_list:
            total_hops = sum(hops_list)
            all_lats = np.random.lognormal(mean=2.0, sigma=0.5, size=total_hops)
            idx = 0
            max_latency = 0.0
            for hops in hops_list:
                lat = all_lats[idx : idx + hops].sum()
                idx += hops
                if lat > max_latency:
                    max_latency = lat
            if max_latency <= L_max:
                C_t = 1
        else:
            C_t = 1
            
        if C_t == 0:
            ttf = step
            break
            
    return ttf

def worker_simulate(arg):
    scenario, run_id = arg
    np.random.seed(42 + run_id)
    return simulate_run_fast(scenario)

def run_sensitivity_analysis(n_runs=5000):
    from multiprocessing import Pool
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8]
    dfs = [3, 5, 10, 30]
    
    results = []
    
    # Gaussian sensitivity
    for r in rhos:
        print(f"Running Sensitivity Gaussian rho={r}...")
        s = {'mode': 'S3', 'copula': 'gaussian', 'rho': r}
        with Pool() as pool:
            ttfs = pool.map(worker_simulate, [(s, i) for i in range(n_runs)])
        mttf = np.mean(ttfs)
        results.append({'Copula': 'Gaussian', 'rho': r, 'df': np.nan, 'MTTF': mttf})
        
    # Student-t sensitivity
    for r in rhos:
        for d in dfs:
            print(f"Running Sensitivity Student-t rho={r}, df={d}...")
            s = {'mode': 'S3', 'copula': 't', 'rho': r, 'df': d}
            with Pool() as pool:
                ttfs = pool.map(worker_simulate, [(s, i) for i in range(n_runs)])
            mttf = np.mean(ttfs)
            results.append({'Copula': 'Student-t', 'rho': r, 'df': d, 'MTTF': mttf})
            
    # Clayton sensitivity
    for r in rhos:
        tau = (2.0 / math.pi) * math.asin(r) if r > 0.0 else 0.0
        theta = (2.0 * tau) / (1.0 - tau) if tau < 1.0 else 100.0
        print(f"Running Sensitivity Clayton rho={r} (theta={theta:.4f})...")
        s = {'mode': 'S3', 'copula': 'clayton', 'rho': theta}
        with Pool() as pool:
            ttfs = pool.map(worker_simulate, [(s, i) for i in range(n_runs)])
        mttf = np.mean(ttfs)
        results.append({'Copula': 'Clayton', 'rho': r, 'df': np.nan, 'MTTF': mttf})
        
    # Gumbel sensitivity
    for r in rhos:
        tau = (2.0 / math.pi) * math.asin(r) if r > 0.0 else 0.0
        theta = 1.0 / (1.0 - tau) if tau < 1.0 else 100.0
        print(f"Running Sensitivity Gumbel rho={r} (theta={theta:.4f})...")
        s = {'mode': 'S3', 'copula': 'gumbel', 'rho': theta}
        with Pool() as pool:
            ttfs = pool.map(worker_simulate, [(s, i) for i in range(n_runs)])
        mttf = np.mean(ttfs)
        results.append({'Copula': 'Gumbel', 'rho': r, 'df': np.nan, 'MTTF': mttf})
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(out_dir, "sensitivity_analysis.csv"), index=False)
    print("Sensitivity analysis complete.")

if __name__ == '__main__':
    scenarios = [
        {'name': 'S1_Spatial', 'mode': 'S1', 'copula': 'indep'},
        {'name': 'S2_Temporal', 'mode': 'S2', 'copula': 'indep'},
        {'name': 'S3_Joint_Indep', 'mode': 'S3', 'copula': 'indep'},
        {'name': 'S3_Joint_Gauss', 'mode': 'S3', 'copula': 'gaussian', 'rho': 0.6},
        {'name': 'S3_Joint_T', 'mode': 'S3', 'copula': 't', 'rho': 0.6, 'df': 3},
        {'name': 'S3_Joint_Clayton', 'mode': 'S3', 'copula': 'clayton', 'rho': 1.39},
        {'name': 'S3_Joint_Gumbel', 'mode': 'S3', 'copula': 'gumbel', 'rho': 1.69}
    ]
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    # Regenerate all scenarios with the updated models and baselines
    for s in scenarios:
        print(f"Running {s['name']}...")
        res = run_experiment(s, n_runs=5000)
        csv_path = os.path.join(out_dir, f"{s['name']}.csv")
        res.to_csv(csv_path, index=False)
        print(f"Done {s['name']}.")
        
    run_sensitivity_analysis(n_runs=5000)
