import numpy as np
import random
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def load_video_chunk(buffer, buffer_offset, video, video_offset, chunk_size) -> (bool, int,np.array):
    finished = False

    if video_offset + chunk_size >= video.shape[0]:
        logging.info("Loaded end of video.")
        chunk_size = video.shape[0] - video_offset - 1
        return True, -1, -1
        finished = True

    if buffer_offset + chunk_size > buffer.shape[0]:
        logging.info("Cannot fit video chunk into buffer, load offset set to 0.")
        buffer_offset = 0

    for i in range(chunk_size):
        buffer[buffer_offset + i] = video[video_offset + i]

    logging.info(f"Loaded frames {video_offset}-{video_offset+chunk_size-1}\t"
                 f" into buffer {buffer_offset}-{buffer_offset+chunk_size-1}, (chunk size {chunk_size})")

    return finished, buffer_offset+chunk_size, video_offset+chunk_size


def load_chunk_from_file()


def chunk_analysis(buffer, buffer_offset, chunk_size, fft_mag_accum, ccc_vector, tau_vector):
    """
    Handles DDM analysis on a single

    :param buffer: main data array [#frame, width, height]
    :param buffer_offset: frame offset for analysis
    :param chunk_size: #frame for analysis chunk
    :param fft_mag_accum: Accumulated fourier-transform data array [tau_index, width, height]
    :param ccc_vector: Number of frames that have contributed to fft_mag_accum
    :param tau_vector: Vector containing tau values
    """

    if buffer_offset + chunk_size > buffer.shape[0]:
        logging.info("Indices outside of buffer range, analysis offset set to 0.")
        buffer_offset = 0

    pixel_count = buffer.shape[1] * buffer.shape[2]
    norm_factor = 1 / pixel_count

    logging.info(f"Analysing buffer frames {buffer_offset}-{buffer_offset+chunk_size-1}")

    for tau_idx, tau in enumerate(tau_vector):
        # Currently only take of each tau value per chunk
        for x in range(5):
            index_1 = buffer_offset + random.randint(0, chunk_size - 1 - tau)  # random is likely a bottle neck

            temp_diff = buffer[index_1] - buffer[index_1 + tau]
            fft_temp_diff = np.fft.fft2(temp_diff) * norm_factor

            fft_mag_accum[tau_idx] += np.abs(fft_temp_diff) * 2
            ccc_vector[tau_idx] += 1

    return buffer_offset+chunk_size


def gen_radius_masks(q_num, width, height):
    radius_mask = np.zeros((q_num, width, height))
    radius_mask_px_count = np.zeros(q_num)

    q_vector = np.linspace(1, width/4, num=q_num)

    for q_index in range(q_num):
        for x in range(width):
            for y in range(height):
                r = np.linalg.norm([x - width / 2, y - height / 2])
                # print(r, q[q_index])
                if 1 <= r / q_vector[q_index] <= 1.05:
                    radius_mask[q_index, x, y] = 1
                    radius_mask_px_count[q_index] += 1
        radius_mask[q_index] = np.fft.fftshift(radius_mask[q_index])

    return radius_mask, q_vector, radius_mask_px_count


def analyse_fft_mag_accum(fft_mag_accum, ccc_vector, radius_masks, q_vector, radius_mask_px_count):
    tau_count = ccc_vector.size
    q_count = q_vector.size

    iqtau = np.zeros((q_count, tau_count))

    for tau_index in range(tau_count):
        fft_mag_accum[tau_index] = fft_mag_accum[tau_index] / ccc_vector[tau_index]

        for q_index in range(q_count):
            if radius_mask_px_count[q_index] == 0:
                iqtau[q_index, tau_index] = np.NaN
            else:
                iqtau[q_index, tau_index] = np.sum(radius_masks[q_index] * fft_mag_accum[tau_index]) \
                                            / radius_mask_px_count[q_index]

    return q_vector, iqtau


def ddm_circ(video_frames, tau_vector):
    SINGLE = np.dtype("single")

    _, video_width, video_height = video_frames.shape
    buffer_size = 60
    chunk_size = 20

    analysis_offset = 0
    load_in_offset = 0
    video_offset = 0
    q_count = 40

    buffer = np.zeros((buffer_size, video_width, video_height), SINGLE)
    fft_mag_accum = np.zeros((tau_vector.size, video_width, video_height), SINGLE)
    ccc_vector = np.zeros(tau_vector.size)

    # Load in two video chunks
    finished, load_in_offset, video_offset = load_video_chunk(buffer, load_in_offset, video_frames, video_offset, chunk_size)
    finished, load_in_offset, video_offset = load_video_chunk(buffer, load_in_offset, video_frames, video_offset, chunk_size)

    while not finished:
        analysis_offset = chunk_analysis(buffer, analysis_offset, chunk_size, fft_mag_accum, ccc_vector, tau_vector)
        finished, load_in_offset, video_offset = load_video_chunk(buffer, load_in_offset, video_frames, video_offset, chunk_size)

    chunk_analysis(buffer, analysis_offset, chunk_size, fft_mag_accum, ccc_vector, tau_vector)
    logging.info("Intial analysis complete.")

    # Analyse
    radius_masks, q_vector, radius_mask_px_count = gen_radius_masks(q_count, video_width, video_height)

    out = analyse_fft_mag_accum(fft_mag_accum, ccc_vector, radius_masks, q_vector, radius_mask_px_count)
    logging.info("Full analysis complete.")
    return out

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    # Playing video from file:
    cap = cv2.VideoCapture("0.5um colloids_Trim.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # note this isn't actually super accurate

    size = min(width, height) / 4 # divide 4 to reduce analysed area - remove for real implementation
    size = int(2**np.floor(np.log2(size)))

    vid_frame_array = np.empty((frames, size, size))

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
            vid_frame_array[true_frame_count] = np.average(frame_buffer[w_start:w_end, h_start:h_end, :],
                                                               axis=2, weights=[0.3, 0.6, 0.11])
            true_frame_count += 1

    print("Frame array created, releasing capture")
    cap.release()
    tau_vec = np.array(range(2, 10))

    q_vector, iqtau = ddm_circ(vid_frame_array, tau_vec)

    for tau in range(1, iqtau.shape[1]):
        # plotting the points
        plt.plot(q_vector, iqtau[:, tau])
        plt.xlabel("qs")
    plt.show()

    for qi in range(iqtau.shape[0]):
        # plotting the points
        plt.plot(tau_vec, iqtau[qi])
        plt.xlabel("Tau")
    plt.show()
