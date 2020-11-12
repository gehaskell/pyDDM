import numpy as np
import random

DOUBLE = np.dtype("double")
SINGLE = np.dtype("single")  # by default float-32

def DDM_core(frame_array, frame_count, max_tau=None, num_frame_couples=None):
    """
    :param frame_array: array (width, height, frame count), pop with px values (single)
                        width / height should be powers of 2
           max_tau: max tau value
           num_frame_couples: must make sure is int or signifier for all frames (e.g. "all" etc)
    :return:
    """

    frame_height, frame_width, _ = frame_array.shape

    # Input check

    # our analysis is dependent on square frame
    if frame_width != frame_height:
        print("Warning: non-square frame - square subsection will be analysed")

    frame_size = min(frame_height, frame_width)
    N_px_frame = frame_size ^ 2                     # pixel count
    fft2_norm_factor = 1 / N_px_frame
    N_frames = frame_count

    if not num_frame_couples:
        num_frame_couples = 10
    if not max_tau:
        max_tau = int(N_frames / 2)

    # General-purpose variables

    max_q = int(frame_size / 2)

    # Distance map for fast radial average

    distance_map = np.zeros([frame_size, frame_size])

    for x in range(frame_size):
        for y in range(frame_size):
            dist = np.sqrt((x - max_q )**2 + (y - max_q)**2)
            distance_map[x, y] = np.round(dist)
    dist_map = np.fft.fftshift(distance_map)
    flat_dist_map = np.ndarray.flatten(distance_map.astype(int))
    dist_counts = np.bincount(flat_dist_map) # count values at each dist

    # Actual DDM Calculation
    #   difference of frames then |FFT|**2, average then radial average

    Iqtau = np.zeros([max_q, max_tau], DOUBLE)
    cccount = np.zeros(max_tau)

    for tau in range(1, max_tau):
        # tau is the difference in frame-indices
        # max_tau is half the number of frames by default

        if num_frame_couples >= 0:
            # We want to average over all frames
            ind_frames = np.arange(N_frames - max_tau)
        else:
            start = random.randint(0, tau-1)
            stop = (N_frames-1) - tau

            ind_frames_1 = np.arange(start, stop, tau, dtype=np.dtype("int"))
            ind_frames_2 = np.array([random.randint(0, stop) for x in range(num_frame_couples)])

            if ind_frames_2.size < ind_frames_1.size:
                ind_frames = ind_frames_2
            else:
                ind_frames = ind_frames_1

        print(f"Tau {tau}/{max_tau}")

        accum_abs_FT_diff_image = np.zeros((frame_size, frame_size), SINGLE)
        ccc = 0

        for i in ind_frames:
            temp_diff = frame_array[:, :, i] - frame_array[:, :, i + tau]
            fft_temp_diff = np.fft.fft2(temp_diff) * fft2_norm_factor
            accum_abs_FT_diff_image += np.abs(fft_temp_diff) * 2
            ccc += 1

        # average on initial times(ie on couple of frames at same lag-time)
        averaged_abs_FT_diff_image = accum_abs_FT_diff_image / ccc
        cccount[tau] = ccc

        radial_binned_fft = np.bincount(flat_dist_map, weights=np.ndarray.flatten(averaged_abs_FT_diff_image))
        oneD_power_spectrum = radial_binned_fft / dist_counts

        Iqtau[:, tau] = oneD_power_spectrum[1:max_q+1]

    return Iqtau


if __name__ == '__main__':
    import cv2
    # Playing video from file:
    cap = cv2.VideoCapture("0.5um colloids_Trim.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # note this isn't actually super accurate


    size = min(width, height) / 4 # REMOVE - just so goes faster
    size = int(2**np.floor(np.log2(size)))

    vid_frame_array = np.empty((size, size, frames))

    w_start, w_end = [int((width + sgn*size)/2) for sgn in [-1, 1]]
    h_start, h_end = [int((height + sgn*size)/2) for sgn in [-1, 1]]

    print(f"Width: {width}, Height: {height}, Frames: {frames}, Analysed Size: {size}")

    true_frame_count = 0
    status = True
    while status:
        if not (true_frame_count % 25): print(f"Frame {true_frame_count} / {frames}")

        status, frame_buffer = cap.read()
        if status:
            # Weighted average over RGB using standard luma conversion weights
            vid_frame_array[:, :, true_frame_count] = np.average(frame_buffer[w_start:w_end, h_start:h_end, :],
                                                               axis=2, weights=[0.3, 0.6, 0.11])
            true_frame_count += 1

    print("Frame array created, releasing capture")
    cap.release()

    iqtau = DDM_core(vid_frame_array, true_frame_count)
