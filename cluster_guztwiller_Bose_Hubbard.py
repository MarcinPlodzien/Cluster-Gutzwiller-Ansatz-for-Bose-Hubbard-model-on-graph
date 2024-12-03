#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:03:17 2024

@author: Marcin Plodzien
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import pandas as pd

# Parameters
n_max = 2  # Maximum occupation number per site
L = 2  # Number of sites in the lattice
U = 1.0  # On-site interaction strength
mu_values = np.linspace(0, 1, 20)  # Range of chemical potentials (mu/U)
J_U_values = np.linspace(0, 0.05, 20)  # Range of J/U (hopping strength relative to U)
neighbours_range = L  # Cluster neighbor range
max_cluster_size = None # None  # Set None for unrestricted cluster size
symmetry_breaking_field = 1e-3  # Small perturbation to break symmetry
max_iterations = 1000  # Maximum iterations for self-consistency
tolerance = 1e-8  # Convergence tolerance

# Bosonic operators
def boson_operators(n_max):
    dim = n_max + 1
    a = np.zeros((dim, dim), dtype=np.complex128)
    for n in range(1, dim):
        a[n - 1, n] = np.sqrt(n)
    a_dagger = a.T.conj()
    n_op = np.dot(a_dagger, a)
    return a, a_dagger, n_op

a, a_dagger, n_op = boson_operators(n_max)

# Kronecker product
def kron_all_dense(operators):
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result

# Initialize clusters
def initialize_clusters(G, neighbours_range, max_cluster_size=None, n_max=3):
    clusters = {}
    cluster_info = []
    cluster_adj_matrices = {}

    for node in G.nodes():
        # Handle the special case where neighbours_range = 0
        if neighbours_range == 0:
            cluster_nodes = [node]
        else:
            # Find neighbors within the specified range
            cluster_nodes = set()
            for dist in range(1, neighbours_range + 1):
                cluster_nodes.update(nx.single_source_shortest_path_length(G, node, cutoff=dist).keys())
            cluster_nodes = list(cluster_nodes)

        # Limit the cluster size if specified
        if max_cluster_size:
            cluster_nodes = cluster_nodes[:max_cluster_size]

        # Store the cluster in the dictionary
        clusters[node] = cluster_nodes

        # Compute cluster properties
        cluster_size = len(cluster_nodes)
        hilbert_space_size = (n_max + 1) ** cluster_size

        # Create a subgraph for the cluster and extract the adjacency matrix
        subgraph = G.subgraph(cluster_nodes)
        adj_matrix = nx.adjacency_matrix(subgraph).toarray()
        cluster_adj_matrices[node] = adj_matrix

        # Append information to the DataFrame
        cluster_info.append({
            "Central Node": node,
            "Cluster Nodes": cluster_nodes,
            "Cluster Size": cluster_size,
            "Cluster Hilbert Space Dimension": hilbert_space_size
        })

    # Create a pandas DataFrame from the cluster information
    cluster_df = pd.DataFrame(cluster_info)

    return clusters, cluster_df, cluster_adj_matrices

# Generate site-wise operators
def generate_site_operators(cluster_nodes, n_max):
    cluster_size = len(cluster_nodes)
    identity = np.eye(n_max + 1, dtype=np.complex128)
    a_ops, a_dagger_ops, n_ops = [], [], []
    for site in range(cluster_size):
        ops = [identity] * cluster_size
        ops[site] = a
        a_ops.append(kron_all_dense(ops))
        ops[site] = a_dagger
        a_dagger_ops.append(kron_all_dense(ops))
        ops[site] = n_op
        n_ops.append(kron_all_dense(ops))
    return a_ops, a_dagger_ops, n_ops

# Construct Hamiltonian for a cluster
def construct_H_cluster(U, mu, J, Phi_cluster, cluster_nodes, adj_matrix, n_max, h):
    cluster_size = len(cluster_nodes)
    dim = (n_max + 1) ** cluster_size
    H = np.zeros((dim, dim), dtype=np.complex128)

    # Generate site-wise operators
    a_ops, a_dagger_ops, n_ops = generate_site_operators(cluster_nodes, n_max)

    # On-site terms
    for i in range(cluster_size):
        H += (U / 2) * n_ops[i] @ (n_ops[i] - np.eye(dim)) - mu * n_ops[i]

    # Hopping terms within the cluster
    for i, site_i in enumerate(cluster_nodes):
        for j, site_j in enumerate(cluster_nodes):
            if adj_matrix[i, j] == 1:  # Use local adjacency matrix
                H -= J * (a_dagger_ops[i] @ a_ops[j] + a_ops[i] @ a_dagger_ops[j])

    # Mean-field contribution from neighboring clusters
    for i in range(cluster_size):
        H -= J * (Phi_cluster * a_dagger_ops[i] + np.conj(Phi_cluster) * a_ops[i])

    # Symmetry-breaking field
    for i in range(cluster_size):
        H -= h * (a_ops[i] + a_dagger_ops[i])

    return H

# Main Workflow
# G = nx.cycle_graph(L)  # Example: Cycle graph with periodic boundary conditions

G = nx.grid_graph(dim = (3, 1))
# Initialize clusters and get adjacency matrices
clusters, clusters_df, cluster_adj_matrices = initialize_clusters(G, neighbours_range, max_cluster_size, n_max)
print("\nCluster Hilbert Space DataFrame:")
print(clusters_df)


# Phase diagram initialization
phase_diagram_superfluid = np.zeros((len(mu_values), len(J_U_values)))
phase_diagram_dominant = np.zeros((len(mu_values), len(J_U_values)))

for mu_idx, mu in enumerate(mu_values):
    for J_U_idx, J_U in enumerate(J_U_values):
        J = J_U * U  # Hopping strength
        print(f"Processing: mu/U = {mu:.2f}, J/U = {J_U:.2f} (Index mu = {mu_idx+1}/{len(mu_values)}, J/U = {J_U_idx+1}/{len(J_U_values)})")

        if len(clusters) == 1:  # ED Limit
            cluster_nodes = list(G.nodes())
            H_C = construct_H_cluster(U, mu, J, 0, cluster_nodes, nx.adjacency_matrix(G).toarray(), n_max, symmetry_breaking_field)
            eigenvalues, eigenvectors = eigh(H_C)
            ground_state = eigenvectors[:, 0]
            ground_state /= np.linalg.norm(ground_state)

            # Compute superfluid order parameter
            superfluid_order = np.sum(
                [np.sqrt(idx) * np.conj(ground_state[idx - 1]) * ground_state[idx] for idx in range(1, len(ground_state))]
            )
            phase_diagram_superfluid[mu_idx, J_U_idx] = np.abs(superfluid_order) / L

            # Compute one-body density matrix and dominant eigenvalue
            rho_cluster = np.zeros((L, L), dtype=np.complex128)
            a_ops, a_dagger_ops, _ = generate_site_operators(cluster_nodes, n_max)
            for i in range(L):
                for j in range(L):
                    rho_cluster[i, j] = ground_state.conj().T @ a_dagger_ops[i] @ a_ops[j] @ ground_state

            rho_eigenvalues = np.linalg.eigvalsh(rho_cluster)
            phase_diagram_dominant[mu_idx, J_U_idx] = np.max(rho_eigenvalues)
            continue

        # Gutzwiller cluster approach
        phi_C = {node: 1e-2 for node in clusters}
        f_C_n = {node: np.ones((n_max + 1) ** len(clusters[node]), dtype=np.complex128) /
                        np.sqrt((n_max + 1) ** len(clusters[node])) for node in clusters}

        for iteration in range(max_iterations):
            phi_C_old = phi_C.copy()
            average_dominant_eigenvalue = 0
            for node, cluster_nodes in clusters.items():
                Phi_cluster = sum(phi_C[neighbor] for neighbor in cluster_nodes if neighbor in phi_C)
                H_C = construct_H_cluster(U, mu, J, Phi_cluster, cluster_nodes, cluster_adj_matrices[node], n_max, symmetry_breaking_field)
                eigenvalues, eigenvectors = eigh(H_C)
                ground_state = eigenvectors[:, 0]
                ground_state /= np.linalg.norm(ground_state)

                # Compute superfluid order parameter
                phi_C[node] = np.sum(np.sqrt(idx) * np.conj(ground_state[idx - 1]) * ground_state[idx]
                                     for idx in range(1, len(ground_state)))

                # Compute one-body density matrix for the cluster
                cluster_size = len(cluster_nodes)
                rho_cluster = np.zeros((cluster_size, cluster_size), dtype=np.complex128)
                a_ops, a_dagger_ops, _ = generate_site_operators(cluster_nodes, n_max)
                for i in range(cluster_size):
                    for j in range(cluster_size):
                        rho_cluster[i, j] = ground_state.conj().T @ a_dagger_ops[i] @ a_ops[j] @ ground_state

                # Store the dominant eigenvalue of the cluster's reduced density matrix
                rho_eigenvalues = np.linalg.eigvalsh(rho_cluster)
                average_dominant_eigenvalue += np.max(rho_eigenvalues)

            # Average over clusters
            average_dominant_eigenvalue /= len(clusters)
            phase_diagram_dominant[mu_idx, J_U_idx] = average_dominant_eigenvalue

            # Check convergence
            max_phi_diff = max(abs(phi_C[node] - phi_C_old[node]) for node in clusters)
            if max_phi_diff < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            superfluid_order = np.mean([abs(phi_C[node]) for node in clusters])
            phase_diagram_superfluid[mu_idx, J_U_idx] = superfluid_order

# Plot the phase diagrams
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Superfluid order parameter phase diagram
im1 = axes[0].imshow(phase_diagram_superfluid, extent=(J_U_values.min(), J_U_values.max(), mu_values.min(), mu_values.max()),
                      origin='lower', aspect='auto', cmap='plasma')
axes[0].set_title('Superfluid Order Parameter <a_i>')
axes[0].set_xlabel('J/U')
axes[0].set_ylabel('μ/U')
fig.colorbar(im1, ax=axes[0], label='<a_i>')

# Dominant eigenvalue phase diagram
im2 = axes[1].imshow(phase_diagram_dominant, extent=(J_U_values.min(), J_U_values.max(), mu_values.min(), mu_values.max()),
                      origin='lower', aspect='auto', cmap='viridis')
axes[1].set_title('Average Dominant Eigenvalue of ρ')
axes[1].set_xlabel('J/U')
axes[1].set_ylabel('μ/U')
fig.colorbar(im2, ax=axes[1], label='Average Dominant Eigenvalue')

plt.tight_layout()
plt.show()
