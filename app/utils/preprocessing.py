import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

def create_sections(file_path, n_clusters):
    """
    Carga los datos de un archivo, agrupa las células en clusters basados en sus posiciones iniciales,
    calcula el porcentaje de células en cada cluster y almacena las envolventes convexas de cada cluster.
    """
    data = np.loadtxt(file_path, delimiter='\t', skiprows=1, usecols=(4, 5, 6))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_

    # Calcular el porcentaje de células en cada cluster
    (unique, counts) = np.unique(labels, return_counts=True)
    percentages = counts / len(labels) * 100
    clusters_percentages = {f"Cluster {i}": f"{percentages[i]:.2f}%" for i in unique}

    _print_cluster_percentages(clusters_percentages)
    _plot_clusters(data, labels)
    hulls = _get_cluster_section(data, labels, n_clusters)

    return hulls

def _print_cluster_percentages(clusters_percentages):
    for cluster, percentage in clusters_percentages.items():
        print(f"{cluster}: {percentage} de las células")

def _plot_clusters(data, labels):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', marker='o')
    ax.set_xlabel('Posición X')
    ax.set_ylabel('Posición Y')
    ax.set_zlabel('Posición Z')
    ax.set_title('Visualización de Clusters 3D')
    plt.show()

def _get_cluster_section(data, labels, n_clusters):
    """
    Calcula y muestra el volumen de la envolvente convexa de cada cluster en 3D y almacena estas envolventes.
    """
    hulls = []
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if len(cluster_points) > 3:
            hull = ConvexHull(cluster_points)
            hulls.append(hull)
        else:
            hulls.append(None)

    for i, hull in enumerate(hulls):
        if hull is not None:
            print(f"Cluster {i}:")
            print(f"  Vértices: {hull.vertices}")
            print(f"  Volumen: {hull.volume:.2f} unidades cúbicas")
            print(f"  Área: {hull.area:.2f} unidades cuadradas")
        else:
            print(f"Cluster {i}: No tiene una envolvente convexa definida.")
    return hulls

def find_sector(point, hulls):
    """
    Determina a qué sector inicial pertenece una ubicación dada.
    """
    for i, hull in enumerate(hulls):
        if hull is not None and in_hull(point, hull):
            return i
    return None

def in_hull(point, hull):
    """
    Verifica si un punto está dentro de la envolvente convexa dada.
    """
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    new_hull = Delaunay(hull.points[hull.vertices])
    return new_hull.find_simplex(point) >= 0
