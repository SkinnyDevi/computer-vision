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

# Eje central vertical compartido entre ambas mitades del rostro
FACE_MIDLINE_IDXS: list[int] = [10, 151, 9, 168, 197, 1, 0, 17, 152]

# Contorno exterior de la mitad derecha e izquierda del sujeto (frente -> mentón)
RIGHT_HALF_FACE_OVAL_IDXS: list[int] = [10, 67, 54, 162, 234, 132, 172, 150, 176, 152]
LEFT_HALF_FACE_OVAL_IDXS: list[int] = [10, 297, 284, 389, 454, 361, 397, 379, 400, 152]

# Nodos estructurales de la malla táctica en media cara (37 puntos por lado)
RIGHT_HALF_FACE_MESH_IDXS: list[int] = [
    *FACE_MIDLINE_IDXS,
    # Contorno exterior (excluyendo los compartidos 10 y 152)
    67, 54, 162, 234, 132, 172, 150, 176,
    # Frente interior
    108, 104,
    # Arco de la ceja
    107, 105, 70,
    # Marco orbital del ojo
    133, 159, 33, 145,
    # Flanco y aleta nasal
    198, 48,
    # Pómulo y mejilla interior
    116, 101, 50, 205, 207,
    # Boca y lateral de mandíbula/mentón
    39, 61, 181, 210,
]

LEFT_HALF_FACE_MESH_IDXS: list[int] = [
    *FACE_MIDLINE_IDXS,
    # Contorno exterior (excluyendo los compartidos 10 y 152)
    297, 284, 389, 454, 361, 397, 379, 400,
    # Frente interior
    337, 333,
    # Arco de la ceja
    336, 334, 300,
    # Marco orbital del ojo
    362, 386, 263, 374,
    # Flanco y aleta nasal
    420, 278,
    # Pómulo y mejilla interior
    345, 330, 280, 425, 427,
    # Boca y lateral de mandíbula/mentón
    269, 291, 405, 430,
]

# Aristas planares de la malla en media cara (órbita ocular y boca abiertas)
RIGHT_HALF_FACE_MESH_CONNECTIONS: list[tuple[int, int]] = [
    # Línea media
    (10, 151), (151, 9), (9, 168), (168, 197), (197, 1), (1, 0), (0, 17), (17, 152),
    # Óvalo exterior
    (10, 67), (67, 54), (54, 162), (162, 234), (234, 132), (132, 172), (172, 150), (150, 176), (176, 152),
    # Frente y sien
    (10, 108), (151, 108), (67, 108), (67, 104), (54, 104), (108, 104),
    (151, 107), (9, 107), (108, 107), (108, 105), (104, 105), (104, 70), (54, 70), (162, 70),
    # Ceja y órbita ocular (centro abierto para no obstruir la retícula)
    (107, 105), (105, 70),
    (168, 107), (168, 133), (107, 133), (107, 159), (105, 159), (105, 33), (70, 33),
    (133, 159), (159, 33), (33, 145), (145, 133),
    (70, 116), (162, 116), (234, 116), (33, 116),
    # Puente, flanco y aleta nasal
    (197, 133), (197, 198), (133, 198), (1, 198), (1, 48), (198, 48),
    # Zona infraorbital y pómulo
    (133, 101), (145, 101), (145, 50), (33, 50), (116, 50), (101, 50),
    (198, 101), (48, 101),
    # Interior de la mejilla
    (116, 132), (50, 132), (132, 205), (50, 205), (101, 205), (48, 205),
    (132, 207), (205, 207), (172, 207),
    # Boca, mejilla inferior y mandíbula
    (1, 39), (0, 39), (48, 39), (48, 61), (39, 61), (205, 61),
    (207, 61), (207, 210), (172, 210), (61, 210), (61, 181), (181, 210),
    (150, 210), (150, 181), (176, 181), (176, 17), (17, 181),
]

LEFT_HALF_FACE_MESH_CONNECTIONS: list[tuple[int, int]] = [
    # Línea media
    (10, 151), (151, 9), (9, 168), (168, 197), (197, 1), (1, 0), (0, 17), (17, 152),
    # Óvalo exterior
    (10, 297), (297, 284), (284, 389), (389, 454), (454, 361), (361, 397), (397, 379), (379, 400), (400, 152),
    # Frente y sien
    (10, 337), (151, 337), (297, 337), (297, 333), (284, 333), (337, 333),
    (151, 336), (9, 336), (337, 336), (337, 334), (333, 334), (333, 300), (284, 300), (389, 300),
    # Ceja y órbita ocular (centro abierto para no obstruir la retícula)
    (336, 334), (334, 300),
    (168, 336), (168, 362), (336, 362), (336, 386), (334, 386), (334, 263), (300, 263),
    (362, 386), (386, 263), (263, 374), (374, 362),
    (300, 345), (389, 345), (454, 345), (263, 345),
    # Puente, flanco y aleta nasal
    (197, 362), (197, 420), (362, 420), (1, 420), (1, 278), (420, 278),
    # Zona infraorbital y pómulo
    (362, 330), (374, 330), (374, 280), (263, 280), (345, 280), (330, 280),
    (420, 330), (278, 330),
    # Interior de la mejilla
    (345, 361), (280, 361), (361, 425), (280, 425), (330, 425), (278, 425),
    (361, 427), (425, 427), (397, 427),
    # Boca, mejilla inferior y mandíbula
    (1, 269), (0, 269), (278, 269), (278, 291), (269, 291), (425, 291),
    (427, 291), (427, 430), (397, 430), (291, 430), (291, 405), (405, 430),
    (379, 430), (379, 405), (400, 405), (400, 17), (17, 405),
]