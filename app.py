import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import atan2, degrees, sqrt, pi

st.set_page_config(
    page_title="PR-Live Pattern Recognition Lab",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 PR-Live: Real-Time Pattern Recognition Lab")
st.caption("Dataset-Free • Model-Free • Interactive Pattern Recognition System")

st.markdown("""
Draw a pattern using the coordinate input below. The system performs
feature extraction, distance-based classification, fuzzy classification,
structural recognition, and unsupervised C-means clustering.
""")

# ---------------------------------------------------------
# Pattern input
# ---------------------------------------------------------
st.sidebar.header("Pattern Input")

pattern = st.sidebar.selectbox(
    "Choose a pattern to generate",
    ["Triangle", "Rectangle", "Circle", "Zigzag", "Star", "Line"]
)

points_count = st.sidebar.slider("Number of points", 10, 80, 40)

def generate_pattern(name, n):
    if name == "Triangle":
        base = np.array([[0, 0], [1, 0], [0.5, 1], [0, 0]])
        return interpolate(base, n)

    if name == "Rectangle":
        base = np.array([
            [0, 0], [1, 0], [1, 0.7],
            [0, 0.7], [0, 0]
        ])
        return interpolate(base, n)

    if name == "Circle":
        t = np.linspace(0, 2*pi, n)
        return np.column_stack((np.cos(t), np.sin(t)))

    if name == "Zigzag":
        x = np.linspace(0, 1, n)
        y = np.where(np.arange(n) % 2 == 0, 0, 1)
        return np.column_stack((x, y))

    if name == "Star":
        t = np.linspace(0, 2*pi, n)
        r = np.where(np.arange(n) % 2 == 0, 1, 0.4)
        return np.column_stack((r*np.cos(t), r*np.sin(t)))

    x = np.linspace(0, 1, n)
    return np.column_stack((x, np.zeros(n)))

def interpolate(points, n):
    d = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative = np.insert(np.cumsum(d), 0, 0)
    target = np.linspace(0, cumulative[-1], n)
    x = np.interp(target, cumulative, points[:, 0])
    y = np.interp(target, cumulative, points[:, 1])
    return np.column_stack((x, y))

points = generate_pattern(pattern, points_count)

# ---------------------------------------------------------
# Plot pattern
# ---------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=points[:, 0],
    y=points[:, 1],
    mode="lines+markers",
    line=dict(width=4),
    marker=dict(size=5)
))

fig.update_layout(
    title="Real-Time Pattern",
    xaxis=dict(range=[-1.3, 1.3], zeroline=False),
    yaxis=dict(range=[-1.3, 1.3], zeroline=False),
    height=500,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------
def extract_features(p):
    dx = np.diff(p[:, 0])
    dy = np.diff(p[:, 1])

    length = np.sum(np.sqrt(dx**2 + dy**2))

    xmin, xmax = np.min(p[:, 0]), np.max(p[:, 0])
    ymin, ymax = np.min(p[:, 1]), np.max(p[:, 1])

    width = xmax - xmin
    height = ymax - ymin

    area = width * height
    aspect = width / (height + 1e-8)

    angles = np.arctan2(dy, dx)
    angle_change = np.diff(np.unwrap(angles))

    curvature = np.mean(np.abs(angle_change))

    start_end = np.linalg.norm(p[0] - p[-1])
    closure = 1 - start_end / (length + 1e-8)

    centroid_x = np.mean(p[:, 0])
    centroid_y = np.mean(p[:, 1])

    return np.array([
        length,
        width,
        height,
        aspect,
        area,
        curvature,
        closure,
        centroid_x,
        centroid_y
    ])

features = extract_features(points)

# ---------------------------------------------------------
# Feature display
# ---------------------------------------------------------
feature_names = [
    "Path Length",
    "Width",
    "Height",
    "Aspect Ratio",
    "Bounding Area",
    "Curvature",
    "Closure",
    "Centroid X",
    "Centroid Y"
]

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Value": np.round(features, 4)
})

st.subheader("🔍 Extracted Pattern Features")
st.dataframe(feature_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Prototype feature vectors
# Generated algorithmically, not from datasets
# ---------------------------------------------------------
prototype_names = [
    "Triangle",
    "Rectangle",
    "Circle",
    "Zigzag",
    "Star",
    "Line"
]

prototype_vectors = []

for name in prototype_names:
    proto = generate_pattern(name, 60)
    prototype_vectors.append(extract_features(proto))

prototype_vectors = np.array(prototype_vectors)

# ---------------------------------------------------------
# Normalized distance classifier
# ---------------------------------------------------------
def normalize_matrix(x):
    minimum = np.min(x, axis=0)
    maximum = np.max(x, axis=0)
    return (x - minimum) / (maximum - minimum + 1e-8)

all_features = np.vstack([prototype_vectors, features])
normalized = normalize_matrix(all_features)

proto_norm = normalized[:-1]
input_norm = normalized[-1]

distances = np.linalg.norm(
    proto_norm - input_norm,
    axis=1
)

best_index = np.argmin(distances)
distance_class = prototype_names[best_index]

# ---------------------------------------------------------
# Fuzzy membership classifier
# ---------------------------------------------------------
similarity = 1 / (1 + distances)
fuzzy_membership = similarity / np.sum(similarity)

fuzzy_index = np.argmax(fuzzy_membership)
fuzzy_class = prototype_names[fuzzy_index]

# ---------------------------------------------------------
# Structural pattern recognition
# ---------------------------------------------------------
def structural_recognition(p):
    f = extract_features(p)

    length = f[0]
    aspect = f[3]
    curvature = f[5]
    closure = f[6]

    if closure > 0.75 and 0.75 < aspect < 1.25:
        if curvature < 0.45:
            return "Circle / Closed Loop"

    if closure > 0.65 and 0.6 < aspect < 1.8:
        return "Closed Polygon"

    if curvature > 0.35 and closure < 0.5:
        return "Zigzag / Irregular Structure"

    if curvature < 0.12:
        return "Straight Line"

    return "General Geometric Pattern"

structural_class = structural_recognition(points)

# ---------------------------------------------------------
# C-Means clustering
# ---------------------------------------------------------
def cmeans(data, clusters=3, iterations=20):
    data = np.asarray(data, dtype=float)

    centers = data[
        np.linspace(0, len(data) - 1, clusters).astype(int)
    ].copy()

    membership = np.zeros((len(data), clusters))

    for _ in range(iterations):
        for i in range(len(data)):
            d = np.linalg.norm(data[i] - centers, axis=1) + 1e-8
            membership[i] = 1 / d
            membership[i] /= np.sum(membership[i])

        for k in range(clusters):
            weights = membership[:, k] ** 2
            centers[k] = np.sum(
                weights[:, None] * data,
                axis=0
            ) / np.sum(weights)

    return centers, membership

cluster_data = proto_norm[:, :2]

centers, memberships = cmeans(cluster_data, 3)

input_membership = 1 / (
    np.linalg.norm(centers - input_norm[:2], axis=1) + 1e-8
)

input_membership /= np.sum(input_membership)

cluster_id = np.argmax(input_membership) + 1

# ---------------------------------------------------------
# Genetic-style feature selection
# ---------------------------------------------------------
def feature_fitness(mask):
    selected = np.where(mask == 1)[0]

    if len(selected) == 0:
        return 0

    variance = np.var(proto_norm[:, selected])
    compactness = len(selected) / len(mask)

    return variance - 0.05 * compactness

population = np.random.randint(
    0, 2, size=(30, len(feature_names))
)

fitness = np.array([
    feature_fitness(individual)
    for individual in population
])

best_gene = population[np.argmax(fitness)]

selected_features = [
    feature_names[i]
    for i in range(len(feature_names))
    if best_gene[i] == 1
]

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
st.subheader("🧩 Pattern Recognition Results")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Distance Classifier",
        distance_class
    )

with c2:
    st.metric(
        "Fuzzy Classifier",
        fuzzy_class
    )

with c3:
    st.metric(
        "Structural Recognition",
        structural_class
    )

st.subheader("🌐 Unsupervised C-Means Analysis")

st.write(
    f"Input pattern belongs most strongly to **Cluster {cluster_id}**."
)

cluster_df = pd.DataFrame({
    "Cluster": [1, 2, 3],
    "Membership": np.round(input_membership, 4)
})

st.dataframe(
    cluster_df,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# Fuzzy membership visualization
# ---------------------------------------------------------
st.subheader("🌫️ Fuzzy Membership")

fuzzy_df = pd.DataFrame({
    "Pattern": prototype_names,
    "Membership": fuzzy_membership
})

st.bar_chart(
    fuzzy_df.set_index("Pattern")
)

# ---------------------------------------------------------
# Genetic feature selection
# ---------------------------------------------------------
st.subheader("🧬 Genetic Feature Selection")

if selected_features:
    st.write(
        "Selected feature subset:",
        ", ".join(selected_features)
    )
else:
    st.write("No feature selected.")

# ---------------------------------------------------------
# Decision explanation
# ---------------------------------------------------------
st.subheader("🧠 Recognition Pipeline")

pipeline = [
    "1. Pattern generation",
    "2. Coordinate acquisition",
    "3. Feature extraction",
    "4. Feature normalization",
    "5. Minimum-distance classification",
    "6. Fuzzy membership classification",
    "7. Structural recognition",
    "8. C-Means unsupervised clustering",
    "9. Genetic feature selection"
]

for step in pipeline:
    st.write(step)

# ---------------------------------------------------------
# Distance table
# ---------------------------------------------------------
st.subheader("📏 Pattern Distance Analysis")

distance_df = pd.DataFrame({
    "Pattern": prototype_names,
    "Distance": np.round(distances, 5),
    "Similarity": np.round(similarity, 5)
})

st.dataframe(
    distance_df.sort_values("Distance"),
    use_container_width=True,
    hide_index=True
)

st.success(
    f"Final detected pattern: {distance_class}"
)

st.caption(
    "PR-Live demonstrates Pattern Recognition concepts without "
    "external datasets, pretrained models, or machine-learning libraries."
)
