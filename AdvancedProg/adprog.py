import math
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x**3 - x - 1

a = float(input("Введите a: "))
b = float(input("Введите b: "))
e = float(input("Введите e: "))

xpoints = []
ypoints = []
dxpoints = []

iteration = 1

while (b - a) / 2 > e:
    x = (a + b) / 2
    if f(x) == 0:
        print("Exact root found.")
        break
    elif f(x) * f(b) < 0:
        a = x
    else:
        b = x
    dx = abs(a - b)
    print(f"Iteration {iteration}: x = {x}, dx = {dx}")
    xpoints.append(iteration)
    ypoints.append(x)
    dxpoints.append(dx)
    iteration += 1

print("xpoints:", xpoints)
print("ypoints:", ypoints)

xPoints = np.array(xpoints)
yPoints = np.array(ypoints)
dxPoints = np.array(dxpoints)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(xPoints, yPoints, marker='o', color='blue')
plt.title('Convergence of Roots')
plt.xlabel('Iterations')
plt.ylabel('Root Value')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(xPoints, dxPoints, marker='o', color='red')
plt.title('Convergence of dx')
plt.xlabel('Iterations')
plt.ylabel('dx')
plt.grid()

plt.tight_layout()
plt.show()
