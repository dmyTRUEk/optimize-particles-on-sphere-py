#

from dataclasses import dataclass

from random import uniform as random_float, randint as random_int
from math import sqrt, exp

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


N: int = 20
SPHERE_RADIUS: float = 1.0

STEPS: int = 10_000
TIME_PER_STEP: float = .1

TEMPERATURE_SCALE: float = .1
TEMPERATURE_TAU  : float = 100

DYNAMIC_CONNECTIONS: bool = True


random_float_m1_1 = lambda: random_float(-1, 1)
random_vec_m1_1 = lambda: np.array([random_float_m1_1(), random_float_m1_1(), random_float_m1_1()])

random_index = lambda: random_int(0, N-1)


def calc_force_attract(r: float) -> float:
	# return 0
	# return r / 10
	# return -r / 10
	return r**2 / 5
	# return -r**2 / 100
	# s=0.5; return (s/r)**12 - (s/r)**6 + 0.1
	# return abs(r-1.1)
	# return .9 / r
	# return (1.1 / r)**2
	# return (.5 / r)**3
	# return (.4 / r)**4

def calc_force_repel(r: float) -> float:
	# return 0
	# return 3 * exp(-(r/1.1)**2)
	return 0.9 / r
	# return (.4 / r)**2
	# return (.5 / r)**3
	# return (.4 / r)**4


@dataclass
class Particle:
	pos: np.ndarray   # Vec<float, 3>
	force: np.ndarray # Vec<float, 3>

	def norm(self) -> float:
		return float(la.norm(self.pos))

	def normalize(self):
		self.pos = self.pos * (SPHERE_RADIUS / self.norm())



particles: list[Particle] = []
for _ in range(N):
	p = Particle(
		pos=random_vec_m1_1(),
		force=None,
	)
	p.normalize()
	particles.append(p)


connected_indices: list[tuple[int, int]]
#connected_indices = [(random_index(), random_index()) for _ in range(random_int(0, N**2))]
connected_indices = []
for i in range(N):
	connected_indices.append((i, i-1))
#	if i % 4 == 0:
#		connected_indices.append((i, i-1))
# 	# if i % 3 > 0:
# 		connected_indices.append((i, i-2))
# 	# connected_indices.append((i, i-3))

for i in range(0, N, 4):

	# connected_indices.append((i, i))
	# connected_indices.append((i, i-1))
	connected_indices.append((i, i-2)) #
	# connected_indices.append((i, i-3))
	# connected_indices.append((i-1, i-3))

	# connected_indices.append((i+1, i))
	# connected_indices.append((i+1, i+1))
	# connected_indices.append((i-1, i-2))
	connected_indices.append((i-1, i-3)) #

	# connected_indices.append((i+2, i))
	# connected_indices.append((i+2, i+1))
	# connected_indices.append((i+2, i+2))
	# connected_indices.append((i+2, i+3))

	# connected_indices.append((i+3, i))
	# connected_indices.append((i+3, i+1))
	# connected_indices.append((i+3, i+2))
	# connected_indices.append((i+3, i+3))

# for i in range(0, N, 5):
# 	connected_indices.append((i, i-1))
# 	connected_indices.append((i-1, i-2))
# 	# connected_indices.append((i-2, i-3))
# 	connected_indices.append((i-3, i-4))
# 	connected_indices.append((i-4, i-5))

# 	connected_indices.append((i, i-2))
	# connected_indices.append((i, i-1))
	# connected_indices.append((i, i-1))
	# connected_indices.append((i, i-2))
	# connected_indices.append((i, i-3))
	# connected_indices.append((i, i-4))
	# connected_indices.append((i, i-5))
	# connected_indices.append((i, i-6))


plt.ion()
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111, projection='3d')
for iteration in range(STEPS):
	print(f"{iteration}")

	for p in particles:
		p.normalize()

	if DYNAMIC_CONNECTIONS:
		# dynamically create connections (connect every particle to three closest)
		connected_indices = []
		for i, p in enumerate(particles):
			index_dist_list = []
			for j, pother in enumerate(particles):
				if i == j: continue
				index_dist_list.append((j, la.norm(p.pos - pother.pos)))
			index_dist_list.sort(key=lambda index_dist: index_dist[1])
			# print(index_dist_list)
			for j in range(3):
				connected_indices.append((i, index_dist_list[j][0]))

	# nullify forces
	for p in particles:
		p.force = np.array([0., 0., 0.])

	# calc forces
	for i in range(len(particles)):
		for j in range(len(particles)):
			# print(f"{i=}, {j=}")
			if i == j: continue
			#assert i != j
			# if i >= j: continue
			#assert i < j
			p1 = particles[i]
			p2 = particles[j]
			r = p1.pos - p2.pos
			# print(r)
			r_len = float(la.norm(r))
			# print(r_len)
			is_connected = ((i, j) in connected_indices) or ((j, i) in connected_indices)
			if is_connected:
				f = calc_force_attract(r_len)
				f_scaled = -f * r
				p1.force += f_scaled
				p2.force -= f_scaled
			f = calc_force_repel(r_len)
			f_scaled = -f * r
			p1.force -= f_scaled
			p2.force += f_scaled

	# remove normal component of the force
	for p in particles:
		dir = p.pos / la.norm(p.pos)
		f_normal_component = (p.force @ dir) * dir
		p.force -= f_normal_component

	# apply forces
	for p in particles:
		p.pos += p.force * TIME_PER_STEP

	# apply temperature
	for p in particles:
		temperature = exp(-iteration / TEMPERATURE_TAU)
		p.pos += TEMPERATURE_SCALE * temperature * random_vec_m1_1()

	for p in particles:
		p.normalize()

	# print(particles)

	# exit(0)
	# Extract coordinates for plotting
	x = [p.pos[0] for p in particles]
	y = [p.pos[1] for p in particles]
	z = [p.pos[2] for p in particles]

	ax.clear()
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("z")
	ax.set_xlim((-1, 1))
	ax.set_ylim((-1, 1))
	ax.set_zlim((-1, 1))
	ax.set_aspect('equal')

	# ax.set_xticks(np.arange(-1, 1, 0.5))
	# ax.set_yticks(np.arange(-1, 1, 0.5))
	# ax.set_zticks(np.arange(-1, 1, 0.5))

	# ax.scatter(x, y, z)
	for p in particles:
		ax.plot(p.pos[0], p.pos[1], p.pos[2])

	for i, j in connected_indices:
		p1 = particles[i].pos
		p2 = particles[j].pos
		ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='b')

	# forces
	for p in particles:
		f = p.pos + p.force * TIME_PER_STEP
		ax.plot([p.pos[0], f[0]], [p.pos[1], f[1]], [p.pos[2], f[2]], color='r')

	plt.pause(0.1)

plt.pause(1e9)

