import numpy as np

RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]   # Ojo derecho del sujeto
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]   # Ojo izquierdo del sujeto
RIGHT_IRIS_IDX = 468
LEFT_IRIS_IDX = 473

MOUTH_CORNERS = (61, 291)
MOUTH_VERT_PAIRS = [(81, 178), (13, 14), (311, 402)]

PNP_LANDMARK_IDXS = [
    1, # Nariz
    152, # Mentón
    263, # Ojo izq. externo
    33, # Ojo der. externo
    291, # Comisura izq.
    61 # Comisura der.
]

# 3D Axis Model Points for head pose estimation
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nariz (1)
    (0.0, 330.0, 65.0),       # Mentón (152)
    (225.0, -170.0, 135.0),   # Ojo izq. externo (263)
    (-225.0, -170.0, 135.0),  # Ojo der. externo (33)
    (150.0, 150.0, 125.0),    # Comisura izq. (291)
    (-150.0, 150.0, 125.0),   # Comisura der. (61)
], dtype=np.float64)