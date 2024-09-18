from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from continuousp.vis.atomic_colors import atomic_colors
from pymatgen.core.structure import Structure


def show_single_structure(structure: Structure) -> go.Figure:
    fig = go.Figure()

    a, b, c = structure.lattice.matrix

    vertices = np.array(
        [
            np.array([0, 0, 0]),
            a,
            b,
            c,
            a + b,
            a + c,
            b + c,
            a + b + c,
        ],
    )

    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]

    edge_x, edge_y, edge_z = [], [], []
    for start, end in edges:
        edge_x.extend([vertices[start][0], vertices[end][0], None])
        edge_y.extend([vertices[start][1], vertices[end][1], None])
        edge_z.extend([vertices[start][2], vertices[end][2], None])

    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line={'color': 'black', 'width': 2},
            showlegend=False,
        ),
    )

    site_coords = defaultdict(lambda: {'x': [], 'y': [], 'z': []})
    for site in structure.sites:
        species = site.species_string
        site_coords[species]['x'].append(site.x)
        site_coords[species]['y'].append(site.y)
        site_coords[species]['z'].append(site.z)

    for species, coords in site_coords.items():
        r, g, b = atomic_colors[species]
        fig.add_trace(
            go.Scatter3d(
                x=coords['x'],
                y=coords['y'],
                z=coords['z'],
                mode='markers',
                marker={'size': 5, 'color': f'rgb({r}, {g}, {b})'},
                name=species,
            ),
        )

    fig.update_layout(
        scene={
            'xaxis': {'title': ''},
            'yaxis': {'title': ''},
            'zaxis': {'title': ''},
            'camera': {'projection': {'type': 'orthographic'}},
            'aspectmode': 'data',
        },
        width=500,
        height=500,
    )

    return fig
