import numpy as np 
import matplotlib.pyplot as plt 
from math import factorial

def evaluate_bezier(points, total):
    n = len(points) - 1
    bezier = lambda t: sum((factorial(n) // (factorial(i) * factorial(n-i)))*t**i * (1-t)**(n-i)*points[i] for i in range(n+1))
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return [coord for coord in zip(new_points[:, 0], new_points[:, 1])]

def two_points_calc(point_a, point_b, y):
    k = (point_a[1] - point_b[1]) / (point_a[0] - point_b[0])
    b = point_a[1] - k * point_a[0]
    x = (y - b) / k
    return x

def coord_x_calc(coords, y_sample):
    x_sample = [-2 for i in range(len(y_sample))]
    coords = sorted(coords, key=lambda x:(x[1]))
    if coords[0][1] > y_sample[-1] or coords[-1][1] < y_sample[0]:
        return x_sample
    id = 1
    for idy in range(len(y_sample)):
        if coords[id-1][1] > y_sample[idy]:
            continue
        while id < len(coords) and coords[id][1] < y_sample[idy]:
            id += 1
        if id == len(coords):
            break
        x = two_points_calc(coords[id-1], coords[id], y_sample[idy])
        x_sample[idy] = int(x)
    return x_sample

points = np.array([[0, 160], [500, 120], [758, 500], [1280, 720]])
coords = np.array(evaluate_bezier(points, 100))
y_sample = [i for i in range(160, 720, 10)]
x_sample = coord_x_calc(coords, y_sample)
print(x_sample)
plt.plot(coords[:,0], coords[:,1], 'b-')
plt.plot(x_sample, y_sample, 'r.')
plt.show()