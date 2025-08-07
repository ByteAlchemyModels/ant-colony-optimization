def run_aco_with_time_windows(distances, time_windows, service_time, n_ants, n_iterations, decay, alpha, beta):
    n_points = len(distances)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_distance = np.inf
    distance_history = []

    for i in range(n_iterations):
        all_paths = []
        for ant in range(n_ants):
            path = [0]  # Always start at depot (assume node 0)
            visited = set(path)
            current_time = time_windows[0][0]  # Start at depot earliest time

            valid = True
            while len(visited) < n_points:
                current_node = path[-1]
                probabilities = []
                candidates = []

                for j in range(n_points):
                    if j not in visited:
                        arrival = current_time + distances[current_node, j]
                        wait = max(0, time_windows[j][0] - arrival)
                        actual_arrival = arrival + wait
                        if actual_arrival > time_windows[j][1]:
                            continue  # Arrival outside window; skip!
                        prob = (pheromone[current_node, j] ** alpha) * ((1.0 / (distances[current_node, j] + 1e-10)) ** beta)
                        candidates.append((j, actual_arrival, wait))
                        probabilities.append(prob)
                if not candidates:
                    valid = False
                    break
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                idx = np.random.choice(len(candidates), p=probabilities)
                next_node, new_time, wait = candidates[idx]
                path.append(next_node)
                visited.add(next_node)
                current_time = new_time + service_time

            # Complete tour (return to depot)
            if valid:
                current_distance = sum(distances[path[j], path[j+1]] for j in range(n_points-1))
                current_distance += distances[path[-1], path[0]]  # Back to depot
                all_paths.append((path, current_distance))

        # Pheromone update (as before)
        pheromone *= decay
        for path, dist in all_paths:
            for j in range(n_points-1):
                pheromone[path[j], path[j+1]] += 1.0 / dist
            pheromone[path[-1], path[0]] += 1.0 / dist

        if all_paths:
            current_shortest_path, current_min_distance = min(all_paths, key=lambda x: x[1])
            if current_min_distance < best_path_distance:
                best_path = current_shortest_path
                best_path_distance = current_min_distance
        distance_history.append(best_path_distance)

    return best_path, best_path_distance, distance_history