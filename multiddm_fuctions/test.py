import numpy as np
frame_size = 9
DOUBLE = np.dtype("double")
max_q = int(frame_size / 2)



# Distance map for fast radial average

distance_map = np.zeros([frame_size, frame_size])



for x in range(frame_size):
    for y in range(frame_size):
        dist = np.sqrt((x - max_q )**2 + (y - max_q)**2) + 1
        distance_map[x, y] = np.round(dist)
distance_map = np.fft.fftshift(distance_map)
flat_map = np.ndarray.flatten(distance_map.astype(int))
dist_counts = np.bincount(flat_map) # count values at each dist

print(distance_map)
print(dist_counts)