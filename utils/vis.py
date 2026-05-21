import plotly.graph_objects as go
import numpy as np

def visualize_tgm_mesh(data_sample, title="3D Triangle Mesh"):
    """
    Accepts a PyTorch Geometric Data object with face attributes
    and renders an interactive solid 3D mesh inside the notebook.
    """
    # 1. Extract vertices [Num_Vertices, 3]
    vertices = data_sample.pos.cpu().numpy()
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # 2. Extract faces [3, Num_Faces] and transpose to [Num_Faces, 3]
    # In Plotly, i, j, k represent the indices of the vertices connecting each triangle
    if not hasattr(data_sample, 'face') or data_sample.face is None:
        raise ValueError("This data sample does not contain mesh face attributes! Ensure you didn't parse it into points.")
        
    faces = data_sample.face.cpu().numpy()
    i, j, k = faces[0, :], faces[1, :], faces[2, :]

    # 3. Configure the 3D solid mesh trace
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.9,
        color='lightblue',           # Base solid surface color
        intensity=z,                 # Adds subtle shade variation based on height
        colorscale='Viridis',        # Visual gradient depth mapping
        showscale=False
    )

    # 4. Viewport Layout adjustments
    layout = go.Layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(visible=False), # Hides grid backgrounds for smooth clarity
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        width=700,
        height=500
    )

    # 5. Render directly in the Colab output cell
    fig = go.Figure(data=[mesh], layout=layout)
    fig.show()


def visualize_point_cloud(points, title="3D Point Cloud"):
    """
    Accepts a PyTorch Geometric Data object (from ModelNet) 
    and renders an interactive 3D scatter plot inside the notebook.
    """
    # 1. Extract the [Num_Points, 3] coordinate tensor and convert to NumPy
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 2. Configure the 3D scatter plot trace
    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=z,             # Color points by their depth/height (Z-axis)
            colorscale='Viridis', # Beautiful gradient map
            opacity=0.8
        )
    )

    # 3. Setup the viewport layout
    layout = go.Layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(visible=False), # Hide the grid lines for a cleaner look
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        width=700,
        height=500
    )

    # 4. Render directly in the Colab output cell
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

