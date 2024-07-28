'''
@author: Muhittin Gokmen
@date: 2024-02-16
'''


import numpy as np

import os, math

import itertools
from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, \
    unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from functools import cmp_to_key
import logging


####################
# Global variables #
####################

logger = logging.getLogger(__name__)
float_tolerance = 1e-7


#################
# Main function #
#################
def main():
    # Example usage:
    RootDir = '/Users/muhittingokmen/Dropbox/CHOP-MEF-data_in/monodic/'
    angles_file_path = f'{RootDir}updated_angles/'
    output_folder = f"{RootDir}keypoints/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # anglesFiles = glob.glob(os.path.join(angles_file_path, '*.csv'))
    # anglesFiles = glob.glob(angles_file_path, '*.csv')
    anglesFiles = sorted(os.listdir(angles_file_path))
    anglesFiles = anglesFiles[3:4]
    for angles_file in sorted(anglesFiles):
        if not angles_file.endswith(".csv"):
            continue
        filename_base = os.path.splitext(angles_file)[0]
        filename_base = filename_base.replace('_augmented', '')
        filename_base = filename_base.replace('_angles', '')
        filename_base = filename_base.replace('_updated', '')

        print("filename_base: ", filename_base)
        input_filename = f"{angles_file_path}{angles_file}"

        if not os.path.exists(f"{output_folder}yaw"):
            os.makedirs(f"{output_folder}yaw")
        if not os.path.exists(f"{output_folder}pitch"):
            os.makedirs(f"{output_folder}pitch")
        if not os.path.exists(f"{output_folder}roll"):
            os.makedirs(f"{output_folder}roll")
        if not os.path.exists(f"{output_folder}plots"):
            os.makedirs(f"{output_folder}plots")

        outputfile_yaw = f"{output_folder}yaw/{filename_base}_yaw_kines.csv"
        outputfile_pitch = f"{output_folder}pitch/{filename_base}_pitch_kines.csv"
        outputfile_roll = f"{output_folder}roll/{filename_base}_roll_kines.csv"
        output_plots_folder = f"{output_folder}plots/"
        plot_file_yaw = f"{output_plots_folder}{filename_base}_yaw_kines.png"
        plot_file_pitch = f"{output_plots_folder}{filename_base}_pitch_kines.png"
        plot_file_roll = f"{output_plots_folder}{filename_base}_roll_kines.png"


        kines = Kines(input_filename)
        kines_yaw, kines_pitch, kines_roll, yaw, pitch, roll = kines.get_kines(input_filename)
        kines.write_kines_to_csv(kines_yaw, kines_pitch, kines_roll, outputfile_yaw, outputfile_pitch, outputfile_roll)
        kines.plot_kines_to_file(kines_yaw, kines_pitch, kines_roll, yaw, pitch, roll, output_plots_folder, plot_file_yaw, plot_file_pitch, plot_file_roll )

class Kines():
    def __init__(self, kine_opts):
        self.kine_opts = kine_opts

    def get_kines(self, x):

        alpha = x[:,1].tolist()
        beta = x[:,0].tolist()
        gamma = x[:,2].tolist()
        fps = round(self.kine_opts['fps'], 1)
        sigma = self.kine_opts['sigma']
        num_intervals = self.kine_opts['num_intervals']
        assumed_blur = self.kine_opts['assumed_blur']
        signal_border_width = self.kine_opts['signal_border_width']
        min_size = self.kine_opts['min_size']
        contrast_threshold = self.kine_opts['contrast_threshold']
        face_no = [self.kine_opts['face_number']] * len(alpha)
        conf_part = [self.kine_opts['conf_part_id']] * len(alpha)
        speaking = [self.kine_opts['speaking_id']] * len(alpha)

        #conf_part = [conf] * len(alpha)
        #speaking = [speak] * len(alpha)

        # print("alpha - main before np.: ", alpha[0:10])
        length_original = int(len(alpha))
        init_part_length = int(self.kine_opts['part_length'])       # 1800
        if length_original < init_part_length:
            part_length = int(init_part_length)
            padding = int(part_length - length_original)
        elif length_original == init_part_length:
            part_length = int(init_part_length)
            padding = int(0)
        else:  # length_original > 1800
            k = length_original // init_part_length
            remainder = length_original % init_part_length
            rem_per_part = math.ceil(remainder / k)
            if rem_per_part <= 200:  # share in parts
                part_length = int(init_part_length + rem_per_part)
                padding = int((k * part_length) - length_original)
            else:
                part_length = init_part_length
                padding = int( (k+1) * part_length - length_original)
        #print("length_original: ", length_original, "part_length: ", part_length, "padding: ", padding)
        if padding > 0:
            zeros_arr = [0.0] * int(padding)
            alpha = np.concatenate((alpha, zeros_arr))
            beta = np.concatenate((beta, zeros_arr))
            gamma = np.concatenate((gamma, zeros_arr))
        part_length = int(part_length)
        # print("alpha - main after np.: ", alpha[0:10])
        #print("part_length:  (hma_kine_detection) ", part_length)
        alpha_keypoints = []
        beta_keypoints = []
        gamma_keypoints = []
        for i in range(0, len(alpha), part_length):
            #print("i: ", i)
            alpha_part = alpha[i:i + part_length]
            beta_part = beta[i:i + part_length]
            gamma_part = gamma[i:i + part_length]
            alpha_part_dog = []
            beta_part_dog = []
            gamma_part_dog = []
            alpha_part = self.GaussianBlur1D(alpha_part, assumed_blur)
            beta_part = self.GaussianBlur1D(beta_part, assumed_blur)
            gamma_part = self.GaussianBlur1D(gamma_part, assumed_blur)
            # !!!

            part_keypoints_alpha, part_dog_alpha = self.computeKeypointsAndDescriptors(alpha_part, length_original, i, sigma, num_intervals, assumed_blur, signal_border_width, min_size, contrast_threshold)
            part_keypoints_beta, part_dog_beta = self.computeKeypointsAndDescriptors(beta_part, length_original, i, sigma, num_intervals, assumed_blur, signal_border_width, min_size, contrast_threshold)
            part_keypoints_gamma, part_dog_gamma = self.computeKeypointsAndDescriptors(gamma_part, length_original, i, sigma, num_intervals, assumed_blur, signal_border_width, min_size, contrast_threshold)

            for keypoint in part_keypoints_alpha:

                location = int(keypoint.location_in_its_octave)
                octave_index = int(keypoint.octave_index)
                signal_index = int(keypoint.signal_index)

                keypoint.response_yaw = round(part_dog_alpha[octave_index][signal_index][location], 4)
                keypoint.response_pitch = round(part_dog_beta[octave_index][signal_index][location], 4)
                keypoint.response_roll = round(part_dog_gamma[octave_index][signal_index][location], 4)
                # print("keypoint.response ", keypoint.response, "keypoint.response_yaw: ", keypoint.response_yaw, "keypoint.response_pitch: ", keypoint.response_pitch, "keypoint.response_roll: ", keypoint.response_roll)
                keypoint.pt = int(round(keypoint.pt + i))
                keypoint.angle = alpha[int(keypoint.pt)]
                if keypoint.pt > length_original:
                    #print("WARNING: keypoint.pt > length_original: ", keypoint.pt, length_original)
                    keypoint.pt = int(round(length_original - 1))
                keypoint.pt_video = int(keypoint.pt)
                keypoint.response = round(keypoint.response, 4)
                keypoint.size = round(keypoint.size, 3)
                keypoint.octave = round(keypoint.octave)
                keypoint.class_id = round(keypoint.class_id)
                keypoint.type = int(round(keypoint.type))
                keypoint.angle = round(keypoint.angle, 4)
                keypoint.response_yaw = round(keypoint.response_yaw, 4)
                keypoint.response_pitch = round(keypoint.response_pitch, 4)
                keypoint.response_roll = round(keypoint.response_roll, 4)
                keypoint.signal_index = int(keypoint.signal_index)
                keypoint.first_frame = max(keypoint.pt_video - round(keypoint.size), 0)
                keypoint.last_frame = min(keypoint.pt_video + round(keypoint.size), length_original)
                keypoint.face_no = face_no[keypoint.pt_video]
                keypoint.conf_part = conf_part[keypoint.pt_video]
                keypoint.speaking = speaking[keypoint.pt_video]
                keypoint.fps = fps
                # print("keypoint.octave : ", keypoint.octave, "keypoint.size", keypoint.size)
            alpha_keypoints.append(part_keypoints_alpha)

            for keypoint in part_keypoints_beta:
                location = int(keypoint.location_in_its_octave)
                octave_index = int(keypoint.octave_index)
                signal_index = int(keypoint.signal_index)

                keypoint.response_yaw = round(part_dog_alpha[octave_index][signal_index][location], 4)
                keypoint.response_pitch = round(part_dog_beta[octave_index][signal_index][location], 4)
                keypoint.response_roll = round(part_dog_gamma[octave_index][signal_index][location], 4)

                keypoint.pt = int(round(keypoint.pt + i))
                if keypoint.pt > length_original:
                    #print("WARNING: keypoint.pt > length_original: ", keypoint.pt, length_original)
                    keypoint.pt = int(round(length_original - 1))
                keypoint.pt_video = int(keypoint.pt)
                keypoint.angle = beta[int(keypoint.pt)]
                keypoint.response = round(keypoint.response, 4)
                keypoint.size = round(keypoint.size, 3)
                if keypoint.size > 500:
                    print("ALARM: - keypoint.pt", keypoint.pt, "keypoint.size: ", keypoint.size, "keypoint.octave", keypoint.octave)
                keypoint.octave = round(keypoint.octave)
                keypoint.class_id = round(keypoint.class_id)
                keypoint.type = int(round(keypoint.type))
                keypoint.angle = round(keypoint.angle, 4)
                keypoint.response_yaw = round(keypoint.response_yaw, 4)
                keypoint.response_pitch = round(keypoint.response_pitch, 4)
                keypoint.response_roll = round(keypoint.response_roll, 4)
                keypoint.signal_index = int(keypoint.signal_index)
                keypoint.first_frame = max(keypoint.pt_video - round(keypoint.size), 0)
                keypoint.last_frame = keypoint.pt_video + round(keypoint.size)
                keypoint.face_no = face_no[keypoint.pt_video]
                keypoint.conf_part = conf_part[keypoint.pt_video]
                keypoint.speaking = speaking[keypoint.pt_video]
                keypoint.fps = fps
                # print("keypoint.octave : ", keypoint.octave, "keypoint.size", keypoint.size)
            beta_keypoints.append(part_keypoints_beta)

            for keypoint in part_keypoints_gamma:
                location = int(keypoint.location_in_its_octave)
                octave_index = int(keypoint.octave_index)
                signal_index = int(keypoint.signal_index)

                keypoint.response_yaw = round(part_dog_alpha[octave_index][signal_index][location], 4)
                keypoint.response_pitch = round(part_dog_beta[octave_index][signal_index][location], 4)
                keypoint.response_roll = round(part_dog_gamma[octave_index][signal_index][location], 4)

                keypoint.pt = int(round(keypoint.pt + i))
                if keypoint.pt > length_original:
                    #print("WARNING: keypoint.pt > length_original: ", keypoint.pt, length_original)
                    keypoint.pt = int(round(length_original - 1))
                keypoint.pt_video = int(keypoint.pt)
                keypoint.angle = gamma[int(keypoint.pt)]
                keypoint.response = round(keypoint.response, 4)
                keypoint.size = round(keypoint.size, 3)
                keypoint.octave = round(keypoint.octave)
                keypoint.class_id = round(keypoint.class_id)
                keypoint.type = round(keypoint.type)
                keypoint.angle = round(keypoint.angle, 4)
                keypoint.first_frame = max(keypoint.pt_video - round(keypoint.size), 0)
                keypoint.last_frame = keypoint.pt_video + round(keypoint.size)
                keypoint.face_no = face_no[keypoint.pt_video]
                keypoint.conf_part = conf_part[keypoint.pt_video]
                keypoint.speaking = speaking[keypoint.pt_video]
                keypoint.fps = fps

                # print("keypoint.octave : ", keypoint.octave, "keypoint.size", keypoint.size)
            gamma_keypoints.append(part_keypoints_gamma)

        alpha = alpha[0:length_original]
        beta = beta[0:length_original]
        gamma = gamma[0:length_original]
        alpha_keypoints = list(itertools.chain(*alpha_keypoints))
        beta_keypoints = list(itertools.chain(*beta_keypoints))
        gamma_keypoints = list(itertools.chain(*gamma_keypoints))
        return  beta_keypoints, alpha_keypoints, gamma_keypoints

    def write_kines_to_csv(self, alpha_keypoints, beta_keypoints, gamma_keypoints, outputfile_yaw, outputfile_pitch, outputfile_roll):
        #  write alpha_keypoints to csv file
        print("writing to csv file", outputfile_yaw)
        file = open(outputfile_yaw, 'w')
        names = ["position", "position_video", "level", "size", "response", "class_id", "type", 'angle',
                 'response_yaw', 'response_pitch', 'response_roll', 'signal_index', 'first_frame', 'last_frame',
                 'face_no', 'conf_part', 'speaking','fps']

        file.write(",".join(names) + "\n")
        for keypoint in alpha_keypoints:
            keypointValues = [keypoint.pt, keypoint.pt_video, keypoint.octave, keypoint.size, keypoint.response,
                              keypoint.class_id, keypoint.type, keypoint.angle, keypoint.response_yaw, keypoint.response_pitch,
                              keypoint.response_roll, keypoint.signal_index, keypoint.first_frame, keypoint.last_frame,
                              keypoint.face_no, keypoint.conf_part, keypoint.speaking, keypoint.fps]
            file.write(",".join(map(str, keypointValues)) + "\n")
        file.close()

        print("writing to csv file", outputfile_pitch)
        file = open(outputfile_pitch, 'w')
        file.write(",".join(names) + "\n")
        for keypoint in beta_keypoints:
            keypointValues = [keypoint.pt, keypoint.pt_video, keypoint.octave, keypoint.size, keypoint.response,
                              keypoint.class_id, keypoint.type, keypoint.angle, keypoint.response_yaw, keypoint.response_pitch,
                              keypoint.response_roll, keypoint.signal_index, keypoint.first_frame, keypoint.last_frame,
                              keypoint.face_no, keypoint.conf_part, keypoint.speaking, keypoint.fps]
            file.write(",".join(map(str, keypointValues)) + "\n")
        file.close()

        print("writing to csv file", outputfile_roll)
        file = open(outputfile_roll, 'w')
        file.write(",".join(names) + "\n")
        for keypoint in gamma_keypoints:
            # keypoint.angle, keypoint.first_frame, keypoint.last_frame)
            keypointValues = [keypoint.pt, keypoint.pt_video, keypoint.octave, keypoint.size, keypoint.response,
                              keypoint.class_id, keypoint.type, keypoint.angle, keypoint.response_yaw,keypoint.response_pitch,
                              keypoint.response_roll, keypoint.signal_index, keypoint.first_frame, keypoint.last_frame,
                              keypoint.face_no, keypoint.conf_part, keypoint.speaking, keypoint.fps]
            file.write(",".join(map(str, keypointValues)) + "\n")
        file.close()
##########################################################
    def plot_kines_to_file(self, alpha_keypoints, beta_keypoints, gamma_keypoints, alpha, beta, gamma, output_plots_folder, outputfile_yaw, outputfile_pitch, outputfile_roll):

            yaw_arr = {'position': [0] * len(alpha), 'octave': [0] * len(alpha), 'size': [0] * len(alpha),
                       'angle': [0] * len(alpha)}
            for keypoint in alpha_keypoints:
                yaw_arr['position'][keypoint.pt] = int(keypoint.pt_video)
                yaw_arr['angle'][keypoint.pt] = round(keypoint.angle, 4)
                yaw_arr['size'][keypoint.pt] = round(keypoint.size / 10., 1)

            pitch_arr = {'position': [0] * len(beta), 'octave': [0] * len(beta), 'size': [0] * len(beta),
                         'angle': [0] * len(beta)}
            for keypoint in beta_keypoints:
                pitch_arr['position'][keypoint.pt] = int(keypoint.pt_video)
                pitch_arr['angle'][keypoint.pt] = round(keypoint.angle, 4)
                pitch_arr['size'][keypoint.pt] = round(keypoint.size / 10., 1)

            roll_arr = {'position': [0] * len(gamma), 'octave': [0] * len(gamma), 'size': [0] * len(gamma),'angle': [0] * len(gamma)}
            for keypoint in gamma_keypoints:
                roll_arr['position'][keypoint.pt] = int(keypoint.pt_video)
                roll_arr['angle'][keypoint.pt] = round(keypoint.angle, 4)
                roll_arr['size'][keypoint.pt] = round(keypoint.size / 10., 1)
            print("writing to csv file", outputfile_yaw)
            self.plot_with_circles_and_points(alpha, yaw_arr['position'], yaw_arr['angle'], yaw_arr['size'], outputfile_yaw, 'Yaw')
            print("writing to csv file", outputfile_pitch)
            self.plot_with_circles_and_points(beta, pitch_arr['position'], pitch_arr['angle'], pitch_arr['size'], outputfile_pitch, 'Pitch')
            print("writing to csv file", outputfile_roll)
            self.plot_with_circles_and_points(gamma, roll_arr['position'], roll_arr['angle'], roll_arr['size'], outputfile_roll, 'Roll')

    def plot_angles_and_keypoints(self, signal, keypoints, output_file, title):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.plot(signal)
        plt.title(f"{title} Angles with Keypoints")
        plt.xlabel('Time (frames)')
        plt.ylabel('Angle (degrees)')
        #put circle at keypoints with radius = size

        plt.plot([keypoint.pt for keypoint in keypoints], [signal[int(keypoint.pt)] for keypoint in keypoints], 'o',  color='red', markersize = 3)
        plt.tight_layout()

        plt.savefig(output_file)
        plt.show()
        plt.close()

    def plot_with_circles_and_points(self, alpha, x_arr, y_arr, radius_arr, output_file, title):
        # Create a scatter plot for points
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.plot(alpha)
        plt.title(f"{title} Angles with Keypoints")
        plt.xlabel('Time (frames)')
        plt.ylabel('Angle (degrees)')

        plt.plot(x_arr, y_arr, 'o', markersize = 3,  color='red')

        # Add circles at each (x, y) point with the specified radius
        for x, y, radius in zip(x_arr, y_arr, radius_arr):
            circle = plt.Circle((x, y), radius, color='r', fill=False)
            plt.gca().add_patch(circle)

        # Set labels and legend
        #plt.xlabel('t')
        #plt.ylabel(title)
        #plt.title(f"{title} Keypoints with their scales")
        #plt.legend()
        plt.tight_layout()

        plt.savefig(output_file)
        # Show the plot
       # plt.show()
        plt.close()


    def computeKeypointsAndDescriptors(self, signal, length_original, i, sigma=1.6, num_intervals=3, assumed_blur=0.5, signal_border_width=5, min_size=32.0, contrast_threshold=0.01):
        """Compute SIFT keypoints and descriptors for an input signal
        """
        signal = array(signal, dtype=float32)
        base_signal = self.generateBaseSignal(signal, sigma, assumed_blur)
        num_octaves = self.computeNumberOfOctaves(len(base_signal), min_size)

        #print(f"num_octaves: {num_octaves},  number of levels: {num_octaves * num_intervals},num_intervals: {num_intervals}, sigma : {sigma}")

        gaussian_kernels = self.generateGaussianKernels(sigma, num_intervals)
        #print("gaussian_kernels: ", gaussian_kernels[0:12])
        gaussian_signals = self.generateGaussiansignals(base_signal, num_octaves, gaussian_kernels)
        #print("gaussian_signals length : ", len(gaussian_signals))
        dog_signals = self.generateDoGsignals(gaussian_signals)

        keypoints = self.findScaleSpaceExtrema(gaussian_signals, dog_signals, length_original, i, num_intervals, sigma, signal_border_width,
                                          contrast_threshold)
        #print("keypoints: ", keypoints[0:10])
        keypoints_updated = self.removeDuplicateKeypoints(keypoints)
        keypoints_updated = self.convertKeypointsToInputsignalSize(keypoints_updated)

        return keypoints_updated, dog_signals


    #########################
    # signal pyramid related #
    #########################

    def generateBaseSignal(self, signal, sigma, assumed_blur):
        """Generate base signal from input signal by upsampling by 2 in time and blurring
        """
        logger.debug('Generating base signal...')
        # resized_signal
        #print("signal.shape in generateBaseSignal: ", signal.shape[0])
        # signal = np.transpose(resize(signal, (int(signal.shape[0] * 2), 1), interpolation=INTER_NEAREST))
        signal = self.zoom_by_2(signal)
        #print("based signal=", signal)

        sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
        #print("signal length in generateBaseSignal after resize: ", len(signal))
        return self.GaussianBlur1D(signal, sigma_diff)  # the signal blur is now sigma instead of assumed_blur

    def zoom_by_2(self, signal):
        output_length = len(signal) * 2
        output = [0.0] * output_length
        for i in range(output_length):
            if i % 2 == 0:
                output[i] = signal[i // 2]
            else:
                last_index = min(len(signal) - 1, i // 2 + 1)
                output[i] = (signal[i // 2] + signal[last_index]) / 2.
        return output
    def downsample_by_2(self, signal):
        output_length = len(signal) // 2
        output = [0.0] * output_length
        for i in range(output_length):
            output[i] = signal[2 * i]
        return output


    def computeNumberOfOctaves(self, signal_shape, min_size=32.0):
        """Compute number of octaves in signal pyramid as function of base signal shape (OpenCV default)
        """
        #print("signal_shape: - computeNumberOfOctaves ", signal_shape)
        # return int(round(log(signal_shape[0]) / log(2) - 1))
        # we want less octaves size to be 16
        no_octaves = int(round(log(signal_shape / min_size) / log(2)))
        #print("no_octaves: ", no_octaves)
        if no_octaves <= 0:
            no_octaves = 1
        return no_octaves


    def generateGaussianKernels(self, sigma, num_intervals):
        """Generate list of gaussian kernels at which to blur the input signal. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
        """
        logger.debug('Generating scales...')
        num_signals_per_octave = num_intervals + 3
        k = 2 ** (1. / num_intervals)
        gaussian_kernels = zeros(
            num_signals_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
        gaussian_kernels[0] = sigma

        for signal_index in range(1, num_signals_per_octave):
            sigma_previous = (k ** (signal_index - 1)) * sigma
            sigma_total = k * sigma_previous
            gaussian_kernels[signal_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
        return gaussian_kernels


    def generateGaussiansignals(self, signal, num_octaves, gaussian_kernels):
        """Generate scale-space pyramid of Gaussian signals
        """
        logger.debug('Generating Gaussian signals...')
        gaussian_signals = []

        for octave_index in range(num_octaves):
            #print("octave_index: ", octave_index)
            gaussian_signals_in_octave = []
            gaussian_signals_in_octave.append(signal)  # first signal in octave already has the correct blur
            for gaussian_kernel in gaussian_kernels[1:]:
                signal = self.GaussianBlur1D(signal, gaussian_kernel)
                gaussian_signals_in_octave.append(signal)
            gaussian_signals.append(gaussian_signals_in_octave)
            octave_base = gaussian_signals_in_octave[-3]
            # signal = np.transpose(np.array(resize(octave_base, (int(octave_base.shape[0] / 2), 1), interpolation=INTER_NEAREST)))
            signal = self.downsample_by_2(octave_base)

        return array(gaussian_signals, dtype=object)

    def GaussianBlur1D(self, input_signal, sigma):
        filter = self.GaussianFilter(sigma)
        output = self.convolution(input_signal, filter)
        return output
    def convolution(self, signal, kernel):
        output = [0.0] * len(signal)
        size = len(kernel)
        halfsize = size // 2
        for i in range(len(signal)):
            sum = 0.0
            for k in range(-halfsize, halfsize + 1):
                index = i + k
                if index < 0:
                    index = abs(index)- 1
                    index = min(index, len(signal) - 1)
                elif index > len(signal) - 1:
                    index =  index - len(signal) + 1
                    index = max (0, len(signal) - index)
                if index < 0 or index > len(signal) - 1:
                    print("index out of range index, signal length : ", index, len(signal))

                sum += signal[index] * kernel[k + halfsize]
            output[i] = sum
        return output

    def GaussianFilter(self, sigma):
        # 1D Gaussian filter
        radius = int(4 * sigma)
        length = 2 * radius + 1
        size = length // 2
        x = range(-size, size+1 )
        g = [exp(-(xi ** 2.0) / (float(size + 0.0001))) for xi in x]
        #g = [exp(-(xi ** 2.0)/(float(int(4. * sigma) + 0.0001))) for xi in x]
        g = g / sum(g)
        return g
    def difference(self, signal1, signal2):
        #Difference two signals
        difference = [signal1[i] - signal2[i] for i in range(len(signal1))]
        return difference
    def generateDoGsignals(self, gaussian_signals):
        """Generate Difference-of-Gaussians signal pyramid
        """
        logger.debug('Generating Difference-of-Gaussian signals...')
        dog_signals = []

        for gaussian_signals_in_octave in gaussian_signals:
            dog_signals_in_octave = []
            for first_signal, second_signal in zip(gaussian_signals_in_octave, gaussian_signals_in_octave[1:]):
                dog_signals_in_octave.append(self.difference(first_signal, second_signal))
            dog_signals.append(dog_signals_in_octave)
        return array(dog_signals, dtype=object)


    ###############################
    # Scale-space extrema related #
    ###############################

    def findScaleSpaceExtrema(self, gaussian_signals, dog_signals, length_original, ind, num_intervals, sigma, signal_border_width, contrast_threshold=0.01):
        """Find pixel positions of all scale-space extrema in the signal pyramid
        """
        logger.debug('Finding scale-space extrema...')
        threshold = contrast_threshold / num_intervals * 50
        keypoints = []

        for octave_index, dog_signals_in_octave in enumerate(dog_signals):
            for signal_index, (first_signal, second_signal, third_signal) in enumerate(
                    zip(dog_signals_in_octave, dog_signals_in_octave[1:],
                        dog_signals_in_octave[2:])):  # this will produce triplets of adjacent signals
                #print("octave_index, signal_index, len dog per octave: ", octave_index, signal_index, len(dog_signals_in_octave))
                # (i, j) is the center of the 3x3 array,   signal index : 0, 1 and 2

                #print("first_signal length -findExtreme: ", len(first_signal))
                for i in range(signal_border_width, len(first_signal) - signal_border_width):
                    # print("first_signal[i - 1 : i + 2]", first_signal[i - 1 : i + 2])
                    extremum, peak = self.isPixelAnExtremum(first_signal[i - 1: i + 2], second_signal[i - 1: i + 2],
                                                       third_signal[i - 1: i + 2], threshold)
                    if extremum:
                        # print(f"extremum found: octave_index: {octave_index}, signal_index: {signal_index}, i: {i}")
                        localization_result = self.localizeExtremumViaQuadraticFit(i, signal_index + 1, octave_index,
                                                                              num_intervals, dog_signals_in_octave, sigma,
                                                                              contrast_threshold, signal_border_width)
                        keypoint, localized_signal_index = localization_result
                        keypoint.octave_index = octave_index
                        keypoint.signal_index = signal_index + 1
                        keypoint.type = peak

                        #print("reponse, and dog response - findScaleSpaceExtrema : ",keypoint.response, dog_signals[keypoint.octave_index][keypoint.signal_index][i])
                        #print("length: ", len(first_signal), "Frame : ", i, "keypoint.pt = ", int(round(keypoint.pt)), "octave_index : ", octave_index, "signal_index = ",  signal_index, "keypoint.octave (level) : ", keypoint.octave, "signal size : ", keypoint.size)
                        if keypoint.pt < 2 * (length_original - ind -1):
                            keypoints.append(keypoint)
                        #print(
                            #f"total no_keypoints: {len(keypoints)}, octave index: {octave_index}, signal_index, {signal_index}, type: {peak}")
            #print(
                #f"total no_keypoints in octave index {octave_index} is {len(keypoints)}")
            # print("keypoints: ", keypoints)
        return keypoints

    def flatten(self, list_of_lists):
        return sum(list_of_lists, [])

    def isPixelAnExtremum(self, first_subsignal, second_subsignal, third_subsignal, threshold):
        """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise
        """
        # print("subsignals - isPixelAnExtremum: ", first_subsignal, second_subsignal, third_subsignal)
        center_pixel_value = second_subsignal[1]
        peak = 0
        if abs(center_pixel_value) > threshold:
            if center_pixel_value > 0:
                peak = 1
                return (all(center_pixel_value >= first_subsignal) and \
                        all(center_pixel_value >= third_subsignal) and \
                        center_pixel_value >= second_subsignal[0] and \
                        center_pixel_value >= second_subsignal[2]), peak
                # peak
            elif center_pixel_value < 0:
                peak = -1
                return (all(center_pixel_value <= first_subsignal) and \
                        all(center_pixel_value <= third_subsignal) and \
                        center_pixel_value <= second_subsignal[0] and \
                        center_pixel_value <= second_subsignal[2]), peak
                # valley
        return False, peak


    def localizeExtremumViaQuadraticFit(self, i, signal_index, octave_index, num_intervals, dog_signals_in_octave, sigma,
                                        contrast_threshold, signal_border_width, eigenvalue_ratio=10,
                                        num_attempts_until_convergence=5):
        """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
        """
        logger.debug('Localizing scale-space extrema...')
        extremum_is_outside_signal = False
        delta_x = 0.0
        signal_shape = len(dog_signals_in_octave[0])
        first_signal = dog_signals_in_octave[signal_index - 1]
        second_signal = dog_signals_in_octave[signal_index]
        third_signal = dog_signals_in_octave[signal_index + 1]
        # pixel_cube = [[first_signal[i - 1 : i + 2]], [second_signal[i - 1 : i + 2]], [third_signal[i - 1 : i + 2]]].astype('float32')
        # pixel_cube = pixel_cube / 255.0
        # quadratic fit and peak valley detection
        signal_value_at_center = second_signal[i]
        # signal_value_at_extrema = signal_value_at_center
        left_pixel_value = second_signal[i - 1]
        center_pixel_value = second_signal[i]
        right_pixel_value = second_signal[i + 1]

        delta_x = float(
            0.5 * (left_pixel_value - right_pixel_value) / (left_pixel_value - 2. * center_pixel_value + right_pixel_value))
        # delta_x = float( min(max(delta_x, -1), 1))  # limit step size by [-1,1] range
        # print("delta_x: ", delta_x)
        i_updated = i + int(delta_x)
        if i_updated < signal_border_width or i > signal_shape - signal_border_width:
            delta_x = 0.0
            extremum_is_outside_signal = True
            peak_magnitude = center_pixel_value
        else:
            peak_magnitude = center_pixel_value - 0.25 * (left_pixel_value - right_pixel_value) * delta_x
        extremum_update = delta_x
        signal_value_at_extrema = peak_magnitude
        # delta_sigma also can be calculated but not used
        #upper_pixel_value = third_signal[i]
        #lower_pixel_value = first_signal[i]
        #center_pixel_value = second_signal[i]
        #delta_sigma = 0.5 * (upper_pixel_value - 2. * center_pixel_value + lower_pixel_value)

        # signal_index += int(round(sigma_update)) if delta_sigma is updated
        keypoint = KeyPoint1D()
        keypoint.pt = (i + extremum_update) * (2 ** (octave_index))
        keypoint.octave = (num_intervals * octave_index) + signal_index
        keypoint.size = round(sigma * (2 ** ((signal_index) / float32(num_intervals)) * (2 ** (octave_index + 1))),4)  # octave_index + 1 because the input signal was doubled
        keypoint.response = round(signal_value_at_extrema,4)
        keypoint.class_id = 0
        keypoint.type = 0
        keypoint.location_in_its_octave = int(round(i + extremum_update))
        return keypoint, signal_index



    ##############################
    # Duplicate keypoint removal #
    ##############################
    def compareKeypoints(self,keypoint1, keypoint2):
        #Return True if keypoint1 is less than keypoint2 MG
        if keypoint1.pt != keypoint2.pt:
            return keypoint1.pt - keypoint2.pt
        if keypoint1.size != keypoint2.size:
            return keypoint2.size - keypoint1.size
        if keypoint1.response != keypoint2.response:
            return keypoint2.response - keypoint1.response
        if keypoint1.octave != keypoint2.octave:
            return keypoint2.octave - keypoint1.octave
        return keypoint2.class_id - keypoint1.class_id

    def removeDuplicateKeypoints(self, keypoints):
        """Sort keypoints and remove duplicate keypoints
        """
        if len(keypoints) < 2:
            return keypoints

        keypoints.sort(key=cmp_to_key(self.compareKeypoints), reverse=True)
        unique_keypoints = []

        for idk in range(len(keypoints)-1):
            keypoint = keypoints[idk]
            if keypoint.pt == -1:
                continue
            else:
                unique_keypoints.append(keypoint)
                for idr in range(idk+1, len(keypoints)):
                    if abs(int(keypoints[idk].pt) - int(keypoints[idr].pt))  <= 2:
                        keypoints[idr].pt = -1

        unique_keypoints.sort(key=cmp_to_key(self.compareKeypoints), reverse=False)
        return unique_keypoints

    #############################
    # Keypoint scale conversion #
    #############################

    def convertKeypointsToInputsignalSize(self, keypoints):
        """Convert keypoint point, size, and octave to input signal size
        """
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.pt = int(round(0.5 * keypoint.pt))
            keypoint.size = round( 0.5 * keypoint.size, 4)
            # keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            keypoint.octave = int(keypoint.octave)
            converted_keypoints.append(keypoint)
        return converted_keypoints

    def read_angles_file(self, input_filename):
        header, columns = self.read_csv(input_filename)
        #print("header: ", header)
        frame_no = list(map(int, columns["frame_no"]))
        face_no = list(map(int, list(map(float, columns["face_no"]))))
        conf_part = list(map(int, list(map(float, columns["conf_part"]))))
        speaking = list(map(int, list(map(float, columns["speaking"]))))
        fps = list(map(float, columns["fps"]))
        yaw = list(map(float, columns["yaw"]))
        pitch = list(map(float, columns["pitch"]))
        roll = list(map(float, columns["roll"]))
        return frame_no, face_no, conf_part, speaking, fps, yaw, pitch, roll

    def read_csv(self, file_path):
        import csv
        columns = {}
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            header = reader.fieldnames
            # Assuming the first row contains column headers
            for row in reader:
                for column, value in row.items():
                    if column not in columns:
                        columns[column] = []
                    columns[column].append(value)
        return header, columns

class KeyPoint1D():
    def __init__(self):
        self.pt = float
        self.pt_video = int
        self.octave = int
        self.size = float
        self.response = float
        self.class_id = None
        self.type = int
        self.angle = float
        self.response_yaw = float
        self.response_pitch = float
        self.response_roll = float
        self.octave_index = int
        self.signal_index = int
        self.location_in_its_octave = int
        self.first_frame = int
        self.last_frame = int
        self.conf_part = int
        self.face_no = int
        self.speaking = int
        self.fps = None


if __name__ == '__main__':
    main()




