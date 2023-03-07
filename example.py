import numpy as np
R = np.array([[0.9961947, -0.08715574, 0.],
              [0.08715574, 0.9961947, 0.],
              [0., 0., 1.]])
theta = np.arccos((np.trace(R) - 1) / 2)
r = (1 / (2 * np.sin(theta))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
k = r / np.linalg.norm(r)
q = np.array([np.cos(theta / 2), k[0] * np.sin(theta / 2), k[1] * np.sin(theta / 2), k[2] * np.sin(theta / 2)])

theta = 2 * np.arccos(q[0])
k = q[1:] / np.linalg.norm(q[1:])
K = np.array([[0, -k[2], k[1]],
              [k[2], 0, -k[0]],
              [-k[1], k[0], 0]])
R = np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

print(R)