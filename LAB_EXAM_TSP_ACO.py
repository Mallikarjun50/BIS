import numpy as np
import random

# ==============================
# Parameters
# ==============================
NUM_ANTS = 20
NUM_ITERATIONS = 25   # ← only 25 iterations
ALPHA = 1.0           # pheromone importance
BETA = 3.0            # heuristic importance
RHO = 0.1             # pheromone evaporation rate
Q = 1.0               # pheromone deposit factor
PENALTY = 1000        # penalty for violating time window

# ==============================
# Problem Definition
# ==============================
# Example cities with (x, y, ready_time, due_time, service_time)
cities = [
    (0, 0, 0, 1000, 0),    # Depot
    (10, 10, 10, 50, 5),
    (20, 5, 20, 60, 5),
    (15, 15, 0, 70, 5),
    (5, 20, 30, 80, 5)
]

n = len(cities)
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            dist_matrix[i][j] = np.hypot(cities[i][0] - cities[j][0],
                                         cities[i][1] - cities[j][1])
        else:
            dist_matrix[i][j] = np.inf

# ==============================
# ACO Helper Functions
# ==============================
def initialize_pheromones(n):
    return np.ones((n, n))

def heuristic_info(dist):
    eta = 1 / (dist + np.finfo(float).eps)
    np.fill_diagonal(eta, 0)
    return eta

def calculate_arrival_time(route, dist_matrix, cities):
    time = 0
    total_distance = 0
    penalty = 0
    for i in range(len(route) - 1):
        current_city = route[i]
        next_city = route[i + 1]
        travel_time = dist_matrix[current_city][next_city]
        arrival = time + travel_time
        ready, due, service = cities[next_city][2], cities[next_city][3], cities[next_city][4]

        # Wait if early
        if arrival < ready:
            arrival = ready

        # Penalize if late
        if arrival > due:
            penalty += PENALTY * (arrival - due)

        time = arrival + service
        total_distance += travel_time

    return total_distance, penalty

def construct_solution(pheromones, eta, dist_matrix, cities):
    n = len(cities)
    route = [0]
    unvisited = list(range(1, n))

    while unvisited:
        i = route[-1]
        probs = []
        for j in unvisited:
            tau = pheromones[i][j] ** ALPHA
            eta_ij = eta[i][j] ** BETA
            probs.append(tau * eta_ij)
        probs = np.array(probs)
        probs /= probs.sum()
        next_city = random.choices(unvisited, weights=probs, k=1)[0]
        route.append(next_city)
        unvisited.remove(next_city)

    route.append(0)  # return to depot
    total_distance, penalty = calculate_arrival_time(route, dist_matrix, cities)
    return route, total_distance, penalty

def update_pheromones(pheromones, ants_routes):
    pheromones *= (1 - RHO)
    for route, dist, penalty in ants_routes:
        delta_tau = Q / (dist + penalty + 1e-6)
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            pheromones[a][b] += delta_tau
            pheromones[b][a] += delta_tau

# ==============================
# Main ACO Loop
# ==============================
def ant_colony_tsp_tw():
    pheromones = initialize_pheromones(n)
    eta = heuristic_info(dist_matrix)
    best_route = None
    best_cost = float('inf')

    for iteration in range(NUM_ITERATIONS):
        ants_routes = []
        for _ in range(NUM_ANTS):
            route, dist, penalty = construct_solution(pheromones, eta, dist_matrix, cities)
            total_cost = dist + penalty
            ants_routes.append((route, dist, penalty))
            if total_cost < best_cost:
                best_cost = total_cost
                best_route = route

        update_pheromones(pheromones, ants_routes)
        print(f"Iteration {iteration+1:02d}/{NUM_ITERATIONS} | Best cost so far: {best_cost:.2f}")

    return best_route, best_cost

# ==============================
# Run Optimization
# ==============================
best_route, best_cost = ant_colony_tsp_tw()
print("\n✅ Optimal Route:", best_route)
print(f"💰 Total Cost (Distance + Penalty): {best_cost:.2f}")
