import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def initialize_population(S_adult, S_kids, num_adults, num_kids):
    """Initialize the population of adults and kids with their susceptibility status."""
    # Adults
    adults = [{'id': f'adult_{i}', 'status': 'S' if random.random() < S_adult else 'R'} for i in range(num_adults)]
    # Kids
    kids = [{'id': f'kid_{i}', 'status': 'S' if random.random() < S_kids else 'R'} for i in range(num_kids)]

    return adults, kids

def assign_kids_to_schools(kids, Ns):
    """Assign each kid to a school randomly."""
    for kid in kids:
        kid['school'] = random.randint(0, Ns - 1)
    return kids


def form_households(adults, kids, p_inh, num_1kid_houses, num_adults, num_kids):
    """Form households by pairing adults and assigning kids to households."""
    # Shuffle adults and form pairs
    random.shuffle(adults)
    adult_pairs = [adults[i:i+2] for i in range(0, num_adults, 2)]
    
    # Shuffle kids
    random.shuffle(kids)
    kid_index = 0
    households = []

    for i, pair in enumerate(adult_pairs):
        house = {'adults': pair, 'kids': []}
        # Determine the number of kids in the household
        num_house_kids = 1 if i < num_1kid_houses else 2

        for _ in range(num_house_kids):
            if kid_index >= num_kids:
                break  # No more kids to assign
            kid = kids[kid_index]

            # Check if we should assign the susceptible kid to this household based on p_inh
            if kid['status'] == 'S' and any(adult['status'] == 'S' for adult in pair):
                if random.random() < p_inh:
                    house['kids'].append(kid)
                    kid_index += 1
                else:
                    kid_index += 1  # Skip this kid
            else:
                house['kids'].append(kid)
                kid_index += 1

        households.append(house)

    # Assign any unassigned kids to households that need kids
    unassigned_kids = kids[kid_index:]
    if unassigned_kids:
        for kid in unassigned_kids:
            for house in households:
                expected_kids = 1 if len(house['kids']) == 0 else 2
                if len(house['kids']) < expected_kids:
                    house['kids'].append(kid)
                    break
    return households


def build_network(adults, kids, households, Ns):
    """Build the network with individuals as nodes and edges representing contacts."""
    G = nx.Graph()
    # Add nodes
    for adult in adults:
        G.add_node(adult['id'], status=adult['status'], type='adult')
    for kid in kids:
        G.add_node(kid['id'], status=kid['status'], type='kid', school=kid['school'])
    # Add household edges
    for house in households:
        # Adults
        if len(house['adults']) == 2:
            G.add_edge(house['adults'][0]['id'], house['adults'][1]['id'], relation='spouse')
        # Kids to parents
        for kid in house['kids']:
            for adult in house['adults']:
                G.add_edge(kid['id'], adult['id'], relation='parent')
        # Siblings
        if len(house['kids']) > 1:
            for i in range(len(house['kids'])):
                for j in range(i+1, len(house['kids'])):
                    G.add_edge(house['kids'][i]['id'], house['kids'][j]['id'], relation='sibling')
    # Add school edges
    schools = {i: [] for i in range(Ns)}
    for kid in kids:
        schools[kid['school']].append(kid['id'])
    for school_kids in schools.values():
        for i in range(len(school_kids)):
            for j in range(i+1, len(school_kids)):
                G.add_edge(school_kids[i], school_kids[j], relation='classmate')
    return G

def initialize_infection(G):
    """Infect a random susceptible individual to start the epidemic."""
    susceptible_nodes = [n for n, attr in G.nodes(data=True) if attr['status'] == 'S']
    if not susceptible_nodes:
        return None  # No susceptible individuals
    initial_infected = random.choice(susceptible_nodes)
    G.nodes[initial_infected]['status'] = 'I'
    return G

def gillespie_SIR(G, beta, nu):
    """Run the Gillespie algorithm for SIR dynamics."""
    time = 0
    times = [time]
    S_history = [sum(1 for n in G.nodes if G.nodes[n]['status'] == 'S')]
    I_history = [sum(1 for n in G.nodes if G.nodes[n]['status'] == 'I')]
    R_history = [sum(1 for n in G.nodes if G.nodes[n]['status'] == 'R')]
    
    while True:
        I_nodes = [n for n, attr in G.nodes(data=True) if attr['status'] == 'I']
        if not I_nodes:
            break  # No more infections

        infection_events = []
        for u in I_nodes:
            for v in G.neighbors(u):
                if G.nodes[v]['status'] == 'S':
                    infection_events.append((u, v))

        infection_rate = beta * len(infection_events)
        recovery_rate = nu * len(I_nodes)
        total_rate = infection_rate + recovery_rate

        if total_rate == 0:
            break  # No events can occur

        dt = np.random.exponential(1 / total_rate)
        time += dt

        if random.random() < (infection_rate / total_rate):
            # Infection event
            if infection_events:
                _, v = random.choice(infection_events)
                G.nodes[v]['status'] = 'I'
        else:
            # Recovery event
            u = random.choice(I_nodes)
            G.nodes[u]['status'] = 'R'

        # Record the number of S, I, R
        S_history.append(sum(1 for n in G.nodes if G.nodes[n]['status'] == 'S'))
        I_history.append(sum(1 for n in G.nodes if G.nodes[n]['status'] == 'I'))
        R_history.append(sum(1 for n in G.nodes if G.nodes[n]['status'] == 'R'))
        times.append(time)
    return times, S_history, I_history, R_history


def analyze_susceptible_clusters(G):
    """Analyze the clusters of susceptible individuals in the network."""
    # Step 1: Extract the subgraph of susceptible individuals
    susceptible_nodes = [n for n, attr in G.nodes(data=True) if attr['status'] == 'S']
    G_susceptible = G.subgraph(susceptible_nodes)
    
    # Step 2: Find connected components
    connected_components = list(nx.connected_components(G_susceptible))
    
    # Step 3: Calculate cluster sizes
    cluster_sizes = [len(component) for component in connected_components]
    
    # Step 4: Identify the size of the giant component
    if cluster_sizes:
        giant_component_size = max(cluster_sizes)
    else:
        giant_component_size = 0
    
    # Step 5: Report the results
    num_clusters = len(cluster_sizes)
    
    return num_clusters, cluster_sizes, giant_component_size

def giant_component(G):
    """Analyze the clusters of susceptible individuals in the network."""
    # Step 1: Extract the subgraph of susceptible individuals
    susceptible_nodes = [n for n, attr in G.nodes(data=True) if attr['status'] == 'S']
    G_susceptible = G.subgraph(susceptible_nodes)
    
    # Step 2: Find connected components
    connected_components = list(nx.connected_components(G_susceptible))
    
    # Step 3: Calculate cluster sizes
    cluster_sizes = [len(component) for component in connected_components]
    
    # Step 4: Identify the size of the giant component
    if cluster_sizes:
        giant_component_size = max(cluster_sizes)
    else:
        giant_component_size = 0
    
    return giant_component_size