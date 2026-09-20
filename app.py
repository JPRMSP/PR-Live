import streamlit as st
import numpy as np
import pandas as pd
from math import pi

st.set_page_config(
    page_title="PR-Live Pattern Recognition Lab",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 PR-Live: Pattern Recognition Digital Lab")
st.caption("Dataset-Free • Model-Free • Real-Time Pattern Recognition")

st.write(
    "Generate a geometric pattern and analyze it using feature extraction, "
    "minimum-distance classification, fuzzy classification, structural "
    "recognition, C-means clustering and genetic feature selection."
)

st.sidebar.header("Pattern Generator")

pattern = st.sidebar.selectbox(
    "Select Pattern",
    ["Triangle", "Rectangle", "Circle", "Zigzag", "Star", "Line"]
)

points_count = st.sidebar.slider(
    "Number of points",
    10,
    100,
    50
)


def interpolate(points, n):
    distance = np.sqrt(
        np.sum(np.diff(points, axis=0) ** 2, axis=1)
    )

    cumulative = np.insert(
        np.cumsum(distance),
        0,
        0
    )

    target = np.linspace(
        0,
        cumulative[-1],
        n
    )

    x = np.interp(
        target,
        cumulative,
        points[:, 0]
    )

    y = np.interp(
        target,
        cumulative,
        points[:, 1]
    )

    return np.column_stack((x, y))


def generate_pattern(name, n):

    if name == "Triangle":
        base = np.array([
            [0, 0],
            [1, 0],
            [0.5, 1],
            [0, 0]
        ])
        return interpolate(base, n)

    if name == "Rectangle":
        base = np.array([
            [0, 0],
            [1, 0],
            [1, 0.7],
            [0, 0.7],
            [0, 0]
        ])
        return interpolate(base, n)

    if name == "Circle":
        t = np.linspace(0, 2 * pi, n)
        return np.column_stack((
            np.cos(t),
            np.sin(t)
        ))

    if name == "Zigzag":
        x = np.linspace(0, 1, n)
        y = np.where(
            np.arange(n) % 2 == 0,
            0,
            1
        )
        return np.column_stack((x, y))

    if name == "Star":
        t = np.linspace(0, 2 * pi, n)
        radius = np.where(
            np.arange(n) % 2 == 0,
            1,
            0.4
        )
        return np.column_stack((
            radius * np.cos(t),
            radius * np.sin(t)
        ))

    x = np.linspace(0, 1, n)

    return np.column_stack((
        x,
        np.zeros(n)
    ))


points = generate_pattern(
    pattern,
    points_count
)

st.subheader("📐 Generated Pattern")

chart_data = pd.DataFrame(
    points,
    columns=["X", "Y"]
)

st.line_chart(
    chart_data,
    x="X",
    y="Y"
)

st.subheader("🔍 Feature Extraction")


def extract_features(p):

    dx = np.diff(p[:, 0])
    dy = np.diff(p[:, 1])

    segment_length = np.sqrt(
        dx ** 2 + dy ** 2
    )

    path_length = np.sum(
        segment_length
    )

    width = (
        np.max(p[:, 0])
        - np.min(p[:, 0])
    )

    height = (
        np.max(p[:, 1])
        - np.min(p[:, 1])
    )

    bounding_area = width * height

    aspect_ratio = (
        width / (height + 1e-8)
    )

    angles = np.arctan2(
        dy,
        dx
    )

    angle_change = np.diff(
        np.unwrap(angles)
    )

    curvature = np.mean(
        np.abs(angle_change)
    )

    start_end_distance = np.linalg.norm(
        p[0] - p[-1]
    )

    closure = (
        1
        - start_end_distance
        / (path_length + 1e-8)
    )

    centroid_x = np.mean(
        p[:, 0]
    )

    centroid_y = np.mean(
        p[:, 1]
    )

    return np.array([
        path_length,
        width,
        height,
        aspect_ratio,
        bounding_area,
        curvature,
        closure,
        centroid_x,
        centroid_y
    ])


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

features = extract_features(points)

feature_table = pd.DataFrame({
    "Feature": feature_names,
    "Value": np.round(features, 4)
})

st.dataframe(
    feature_table,
    use_container_width=True,
    hide_index=True
)

st.subheader("🎯 Minimum Distance Classifier")

prototype_names = [
    "Triangle",
    "Rectangle",
    "Circle",
    "Zigzag",
    "Star",
    "Line"
]

prototype_features = []

for name in prototype_names:

    prototype = generate_pattern(
        name,
        60
    )

    prototype_features.append(
        extract_features(prototype)
    )

prototype_features = np.array(
    prototype_features
)


def normalize(data):

    minimum = np.min(
        data,
        axis=0
    )

    maximum = np.max(
        data,
        axis=0
    )

    return (
        data - minimum
    ) / (
        maximum - minimum + 1e-8
    )


combined = np.vstack([
    prototype_features,
    features
])

normalized = normalize(
    combined
)

prototype_normalized = normalized[:-1]
input_normalized = normalized[-1]

distances = np.linalg.norm(
    prototype_normalized
    - input_normalized,
    axis=1
)

best_index = np.argmin(
    distances
)

distance_class = prototype_names[
    best_index
]

distance_table = pd.DataFrame({
    "Pattern": prototype_names,
    "Distance": np.round(
        distances,
        5
    )
})

st.dataframe(
    distance_table.sort_values(
        "Distance"
    ),
    use_container_width=True,
    hide_index=True
)

st.success(
    f"Minimum-distance result: {distance_class}"
)

st.subheader("🌫️ Fuzzy Pattern Classification")

similarity = (
    1 / (1 + distances)
)

membership = (
    similarity
    / np.sum(similarity)
)

fuzzy_table = pd.DataFrame({
    "Pattern": prototype_names,
    "Membership": np.round(
        membership,
        4
    )
})

st.dataframe(
    fuzzy_table,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    fuzzy_table.set_index(
        "Pattern"
    )
)

fuzzy_class = prototype_names[
    np.argmax(membership)
]

st.info(
    f"Highest fuzzy membership: {fuzzy_class}"
)

st.subheader("🧩 Structural Pattern Recognition")


def structural_recognition(p):

    f = extract_features(p)

    aspect = f[3]
    curvature = f[5]
    closure = f[6]

    if (
        closure > 0.75
        and 0.75 < aspect < 1.25
        and curvature < 0.45
    ):
        return "Circle / Closed Loop"

    if (
        closure > 0.65
        and 0.6 < aspect < 1.8
    ):
        return "Closed Polygon"

    if (
        curvature > 0.35
        and closure < 0.5
    ):
        return "Zigzag / Irregular Structure"

    if curvature < 0.12:
        return "Straight Line"

    return "General Geometric Pattern"


structural_result = structural_recognition(
    points
)

st.success(
    f"Structural recognition: {structural_result}"
)

st.subheader("🌐 C-Means Unsupervised Classification")


def cmeans(data, clusters=3, iterations=20):

    centers = data[
        np.linspace(
            0,
            len(data) - 1,
            clusters
        ).astype(int)
    ].copy()

    membership = np.zeros(
        (len(data), clusters)
    )

    for _ in range(iterations):

        for i in range(len(data)):

            distance = (
                np.linalg.norm(
                    data[i] - centers,
                    axis=1
                )
                + 1e-8
            )

            membership[i] = (
                1 / distance
            )

            membership[i] /= np.sum(
                membership[i]
            )

        for k in range(clusters):

            weights = (
                membership[:, k] ** 2
            )

            centers[k] = (
                np.sum(
                    weights[:, None] * data,
                    axis=0
                )
                / np.sum(weights)
            )

    return centers, membership


cluster_data = prototype_normalized[:, :2]

centers, cluster_membership = cmeans(
    cluster_data,
    3
)

input_cluster_membership = 1 / (
    np.linalg.norm(
        centers
        - input_normalized[:2],
        axis=1
    ) + 1e-8
)

input_cluster_membership /= np.sum(
    input_cluster_membership
)

cluster_result = (
    np.argmax(
        input_cluster_membership
    ) + 1
)

cluster_table = pd.DataFrame({
    "Cluster": [1, 2, 3],
    "Membership": np.round(
        input_cluster_membership,
        4
    )
})

st.dataframe(
    cluster_table,
    use_container_width=True,
    hide_index=True
)

st.success(
    f"Input belongs most strongly to Cluster {cluster_result}"
)

st.subheader("🧬 Genetic Feature Selection")


def feature_fitness(mask):

    selected = np.where(
        mask == 1
    )[0]

    if len(selected) == 0:
        return 0

    variance = np.var(
        prototype_normalized[:, selected]
    )

    penalty = (
        0.05
        * len(selected)
        / len(mask)
    )

    return variance - penalty


population = np.random.randint(
    0,
    2,
    size=(30, len(feature_names))
)

fitness = np.array([
    feature_fitness(individual)
    for individual in population
])

best_gene = population[
    np.argmax(fitness)
]

selected_features = [
    feature_names[i]
    for i in range(len(feature_names))
    if best_gene[i] == 1
]

if selected_features:

    st.write(
        "Selected features:",
        ", ".join(selected_features)
    )

else:

    st.write(
        "No feature selected."
    )

st.subheader("🧠 Complete Recognition Pipeline")

pipeline = [
    "Pattern generation",
    "Feature extraction",
    "Feature normalization",
    "Minimum-distance classification",
    "Fuzzy classification",
    "Structural recognition",
    "C-means clustering",
    "Genetic feature selection"
]

for number, step in enumerate(
    pipeline,
    start=1
):
    st.write(
        f"**{number}.** {step}"
    )

st.subheader("🏁 Final Analysis")

result_table = pd.DataFrame({
    "Method": [
        "Minimum Distance",
        "Fuzzy Classification",
        "Structural Recognition",
        "C-Means"
    ],
    "Result": [
        distance_class,
        fuzzy_class,
        structural_result,
        f"Cluster {cluster_result}"
    ]
})

st.dataframe(
    result_table,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "PR-Live implements Pattern Recognition algorithms "
    "directly without external datasets or pretrained models."
)
