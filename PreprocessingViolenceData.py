import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_heat_map(cap):
    ret, first_frame = cap.read()
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255
    num_frames = 1
    sum_heat_map = np.zeros(shape=(mask.shape[0], mask.shape[1]))
    while (cap.isOpened()):
        num_frames = num_frames + 1
        ret, frame = cap.read()
        if (not ret): break
        #cv.imshow("input", frame)
        if (num_frames == 65):
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        sum_heat_map = sum_heat_map + mask[..., 2]
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        #cv.imshow("dense optical flow", rgb)
        prev_gray = gray
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    return sum_heat_map

def crop_image_utility(x, winsize):
    crop_img = (x - (winsize / 2))
    if (crop_img < 0):
        overflow = np.absolute(crop_img)
        crop1 = (x - (winsize / 2 - overflow))
        crop2 = (x + (winsize / 2 + overflow))
    else:
        crop1 = (x - (winsize / 2))
        crop2 = (x + (winsize / 2))
    return crop1, crop2

def get_max_intensity(sum_heat_map):
    coordinates = np.where(sum_heat_map == np.amax(sum_heat_map))
    mid_x = (coordinates[0])[0]
    mid_y = (coordinates[1])[0]
    window_size = 224

    crop_x1, crop_x2 = crop_image_utility(mid_x,window_size)
    crop_y1, crop_y2 = crop_image_utility(mid_y,window_size)

    cropped_template = np.zeros(shape=(int(crop_x2 - crop_x1), int(crop_y2 - crop_y1)))

    # new code
    l,r = getTopLeftCoordinate(sum_heat_map)
    if l == -1 and r == -1:
        return l,r,cropped_template
    # new code

    u = int(l)
    for i in range(window_size):
        v = int(r)
        for j in range(window_size):
            cropped_template[i][j] = sum_heat_map[u][v]
            v = v + 1
        u = u + 1
    return l,r,cropped_template

def withinFrame(i, j, n, m):
    if i < n and j < m and i >=0 and j>=0:
        return True
    return False

def getTopLeftCoordinate(sum_heat_map):
    n = sum_heat_map.shape[0]
    m = sum_heat_map.shape[1]
    sz = 224
    cum_heat_map = np.zeros(shape= [n,m])
    cum_heat_map[0][0] = sum_heat_map[0][0]
    for i in range(1, m):
        cum_heat_map[0][i] = cum_heat_map[0][i-1] + sum_heat_map[0][i]
    for i in range(1, n):
        cum_heat_map[i][0] = cum_heat_map[i-1][0] + sum_heat_map[i][0]
    for i in range(1, n):
        for j in range(1, m):
            cum_heat_map[i][j] = cum_heat_map[i-1][j] + cum_heat_map[i][j-1] - cum_heat_map[i-1][j-1] + sum_heat_map[i][j]

    curr_max = 0
    indx = (-1,-1)
    for i in range(n):
        for j in range(m):
            if not withinFrame(i+ sz-1,j + sz-1, n, m):
                continue
            sum_val = cum_heat_map[i+sz-1][j+sz-1]
            if withinFrame(i+sz-1, j-1, n , m):
                sum_val -= cum_heat_map[i+sz-1][j-1]
            if withinFrame(i-1, j+sz-1, n, m):
                sum_val -= cum_heat_map[i-1][j+sz-1]
            if withinFrame(i-1, j-1, n, m):
                sum_val += cum_heat_map[i-1][j-1]
            if sum_val > curr_max:
                curr_max = sum_val
                indx = (i, j)
    return indx

def process_img(intensity_img,l,r):
    window_size = 224
    crop_img = np.zeros(shape=(window_size, window_size))
    u = int(l)
    for i in range(window_size):
        v = int(r)
        for j in range(window_size):
            crop_img[i][j] = intensity_img[u][v]
            v = v + 1
        u = u + 1
    return crop_img

def crop_videos(l,r,cap):
    tensor_input = np.zeros(shape=(2,4,224,224))
    ret, first_frame = cap.read()
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255
    num_frames = 1
    while (cap.isOpened()):
        num_frames = num_frames + 1
        ret, frame = cap.read()
        if (not ret): break
        rgb_resize_img = cv.resize(frame, (224, 224), interpolation=cv.INTER_AREA)

        # ----visualize the resized RGB frame------
        # plt.imshow(rgb_resize_img)
        # plt.show()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        intensity_img = mask[..., 2]
        crop_img = process_img(intensity_img,l,r)

        # ----visualize the cropped frame------
        # plt.imshow(crop_img)
        # plt.show()

        tensor_input[num_frames - 2][0] = rgb_resize_img[..., 0]
        tensor_input[num_frames - 2][1] = rgb_resize_img[..., 1]
        tensor_input[num_frames - 2][2] = rgb_resize_img[..., 2]
        tensor_input[num_frames - 2][3] = crop_img
    return tensor_input

if __name__ == '__main__':
    dir_list = ["RWF-2000-C-Vid/train/Fight/", "RWF-2000-C-Vid/train/NonFight/", "RWF-2000-C-Vid/val/Fight/","RWF-2000-C-Vid/val/NonFight/"]
    dir_list_out = ["RWF-2000-CroppedOpticalFlow/train/Fight/", "RWF-2000-CroppedOpticalFlow/train/NonFight/", "RWF-2000-CroppedOpticalFlow/val/Fight/", "RWF-2000-CroppedOpticalFlow/val/NonFight/"]
    for k, src in enumerate(dir_list):
        video_list = os.listdir(src)
        i = 1
        for video in video_list:
            if (video == ".DS_Store"):  # ignoring the hidden file
                continue
            video = dir_list[k] + video
            cap = cv.VideoCapture(video)
            sum_heat_map = generate_heat_map(cap)
            l,r, cropped_template = get_max_intensity(sum_heat_map)  # get the topmost left index for cropping
            nzero_count = np.count_nonzero(cropped_template)
            if nzero_count == 0:
                continue

            # ------uncomment to view visualization for maximum intensity map and the corresponding cropped region------
            # plt.subplot(221)
            # plt.imshow(sum_heat_map)
            # plt.title('Frame 1')
            # plt.axis('off')
            # plt.subplot(222)
            # plt.imshow(cropped_template)
            # plt.title('Frame 2')
            # plt.axis('off')
            # plt.show()

            # -----cropping videos frame by frame for max optical flow intensity------
            cap_new = cv.VideoCapture(video)
            tensor_input = crop_videos(l,r,cap_new)
            # saving the 4D tensor (frame * channels * w * h)
            np.save(dir_list_out[k] + "cropped_optical_flow" + str(i), tensor_input)
            i = i + 1

            # -------- validating whether the saved file and the file we load are same or not --------
            # readFile = np.load("/RWF-2000-CroppedOpticalFlow/train/Fight/cropped_optical_flow1.npy")
            # print(np.array_equal(readFile,tensor_input))