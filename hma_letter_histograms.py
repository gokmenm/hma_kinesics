'''
@author: Muhittin Gokmen
@date: 2024-02-16
'''
import csv, os, sys
import numpy as np
import matplotlib.pyplot as plt

class LetterHistograms():
    def __init__(self, hist_opts, kwargs):
        self.hist_opts = hist_opts
        self.kwargs = kwargs
        self.kineme_type = hist_opts['kineme_type']
        self.kineme_types_mapping = {
            'single_letters': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                               's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
            'singleton': ['i', 'r', 'u', 'x', 'y', 'z'],
            'one_sweep': ['u', 'x'],
            'two_sweep': ['uu', 'ux', 'xu', 'xx'],
            'three_sweep': ['uuu', 'uux', 'uxu', 'uxx', 'xuu', 'xux', 'xxu', 'xxx'],
            'one_nod': ['i', 'r'],
            'two_nod': ['ii', 'ir', 'ri', 'rr'],
            'three_nod' : ['iii', 'iir', 'iri', 'irr', 'rii', 'rir', 'rri', 'rrr'],
        }

    def get_segments(self, letter, segment_size = 500):
        letter_segments = []
        for l in range(len(letter)):
            letter_segments.append(letter[l:l + segment_size])
        return letter_segments

    def get_number_of_segments(self, video_length, segment_size = 500):
        if video_length % int(segment_size) > 0:
            number_of_segments = video_length // segment_size + 1
        else:
            number_of_segments = video_length // segment_size
        return int(number_of_segments)
    def get_one_segment(self, segments, index, number_of_segments):
        if index < number_of_segments:
            return segments[index]
        else:
            #print("index out of range")
            return segments[-1]

    def set_histogram_type(self, histogram_type = 'letter_levels'): # or 'letter_combined' or 'levels_and_combined' ):
        if histogram_type == 'letter_levels':
            self.histogram_type = 'letter_levels'
        elif histogram_type == 'letter_combined':
            self.histogram_type = 'letter_combined'
        elif histogram_type == 'letter_levels_and_combined':
            self.histogram_type = 'letter_levels_and_combined'
        else:
            print("Please select a valid choice")
    def get_histogram_type(self):
        return self.histogram_type

    def select_levels(self, choice = 'primes' or 'last_scales' or 'all'):
        if choice == 'primes':
            self.levels = [1,4,7,10,13,16,19]
        elif choice == 'last_scales':
            self.levels = [1,4,7,10,13,16,19,22,25]
        elif choice == 'all':
            self.levels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        else:
            print("Please select a valid choice")
    def set_levels(self, levels):
        self.levels = levels
        return self.levels
    def get_letter_dictionary(self, choice = 'only_letters' or 'all'):
        if choice == 'only_letters':
            self.letter_dictionary = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
                            'p','q','r','s','t','u','v','w','x','y','z']
        elif choice == 'all':
                self.letter_dictionary = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
                            'p','q','r','s','t','u','v','w','x','y','z', '_']
        else:
            print("Please select a valid choice")
        return self.letter_dictionary
    def set_letter_dictionary(self, letter_dictionary):
        self.letter_dictionary = letter_dictionary
        return self.letter_dictionary
    def get_level_histograms(self, letter, level, magnitude, selected_levels, letter_dictionary, hist_value = "magnitude"):

        histogram = np.zeros((len(selected_levels), len(letter_dictionary)), dtype=np.float32)
        n_levels = len(selected_levels)
        n_letter_dictionary = len(letter_dictionary)
        #print("levels: ", selected_levels)
        #print("letter_dictionary: ", letter_dictionary)
        #print("letter: ", letter)
        #print("magnitude: ", magnitude)

        for l in range(len(letter)):
            this_level = int(level[l])
            this_letter = letter[l]
            for letter_id in range(len(letter_dictionary)):
                if this_letter == letter_dictionary[letter_id]:
                    if this_level in selected_levels and this_letter in letter_dictionary:
                        for idl in range(len(selected_levels)):
                            if this_level == selected_levels[idl]:
                                if hist_value == "magnitude":
                                    histogram[idl][letter_id] += round(abs(magnitude[l]), 4)
                                elif hist_value == "count":
                                    histogram[idl][letter_id] += 1.0

        return histogram
    def get_histogram1D(self, histogram, letter, level, magnitude, selected_levels, letter_dictionary):
        n_levels = len(selected_levels)
        n_letter_dictionary = len(letter_dictionary)
        histogram1D = [0.0] * n_letter_dictionary
        for letter_index in range(n_letter_dictionary):
            sum_letter = 0.0
            for idl in range(n_levels):
                sum_letter += histogram[idl][letter_index]
            histogram1D[letter_index] = round(sum_letter, 4)
        return histogram1D
    def combine_histograms(self,  histograms, histogram1D, selected_levels, letter_dictionary):
        n_levels = len(selected_levels)
        n_letter_dictionary = len(letter_dictionary)
        histogram_combined = np.zeros((n_levels + 1, n_letter_dictionary), dtype=np.float32)
        for idl in range(n_levels):
            histogram_combined[idl] = histograms[idl]
        histogram_combined[n_levels] = histogram1D
        return histogram_combined

    def print_level_histograms(self, histogram, selected_levels, letter_dictionary, segment_id):
        np.set_printoptions(precision=4)
        #print("Segment: ", segment_id)
        #for idl in range(len(selected_levels)):
            #print(f'Level {selected_levels[idl]} : {histogram[idl]}')
    def plot_histograms(self,HistFolder, filenameBase, filenameEndLevels, filenameEndCombined, filenameEndBoth, histogram, selected_levels, letter_dictionary, segment_id, histogram_type = 'letter_levels'):
        if not os.path.exists(HistFolder):
            os.makedirs(HistFolder)
        if histogram_type == 'letter_levels' or histogram_type == 'kineme_levels':
            n_plots = len(selected_levels)
            n_rows = (n_plots + 2) // 3  # Add 2 and use integer division to calculate the number of rows
            n_cols = min(n_plots, 3)
            # Creating subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4))  # Adjust the figure size as needed
            axes = np.ravel(axes)
            for idl in range(len(selected_levels)):
                #positions = np.arange(len(self.letter_dictionary))
                ax = axes[idl]
                positions = np.array(letter_dictionary)
                hist = np.array(histogram[idl])
                ax.bar(positions, hist , color='blue', edgecolor='black')
                # Adding labels and title
                #ax.xlabel('letter_dictionary')
                #ax.ylabel('magnitude')
                ax.set_title(f'Histogram of letters at level {selected_levels[idl]}')
            plt.tight_layout()
            if self.hist_opts['save_histogram_plots']:
                plt.savefig(f'{HistFolder}{filenameBase}_{filenameEndLevels}_{segment_id}.png')
            if self.hist_opts['show_histogram_plots']:
                plt.show()
        elif histogram_type == 'letter_combined' or histogram_type == 'kineme_combined':
            n_plots = 1
            n_rows = (n_plots + 2) // 3
            n_cols = min(n_plots, 3)
            # Creating subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4))  # Adjust the figure size as needed
            axes = np.ravel(axes)
            ax = axes[0]
            positions = np.array(letter_dictionary)
            hist = np.array(histogram)
            ax.bar(positions, hist , color='blue', edgecolor='black')
            ax.set_title(f'Histogram of letters at all levels')
            plt.tight_layout()
            if self.hist_opts['save_histogram_plots']:
                plt.savefig(f'{HistFolder}{filenameBase}_{filenameEndCombined}_{segment_id}.png')
            if self.hist_opts['show_histogram_plots']:
                plt.show()

        elif  histogram_type == 'letter_levels_and_combined' or histogram_type == 'kineme_levels_and_combined':
            n_plots = len(selected_levels) + 1
            n_rows = (n_plots + 2) // 3
            n_cols = min(n_plots, 3)
            # Creating subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4))  # Adjust the figure size as needed
            axes = np.ravel(axes)

            for idl in range(len(selected_levels) + 1):
                #positions = np.arange(len(self.letter_dictionary))
                ax = axes[idl]
                positions = np.array(letter_dictionary)
                hist = np.array(histogram[idl])
                ax.bar(positions, hist , color='blue', edgecolor='black')
                # Adding labels and title
                #ax.xlabel('letter_dictionary')
                #ax.ylabel('magnitude')
                if idl < len(selected_levels):
                    ax.set_title(f'Histogram of letters at level {selected_levels[idl]}')
                else:
                    ax.set_title(f'Histogram of letters at all levels')
                #plt.savefig(f'histogram_level_{self.levels[idl]}.png')
            plt.tight_layout()
            plt.title(f'Histograms of segment {segment_id}')
            if self.hist_opts['save_histogram_plots']:
                plt.savefig(f'{HistFolder}{filenameBase}_{filenameEndBoth}_{segment_id}.png')
            if self.hist_opts['show_histogram_plots']:
                plt.show()
        else:
            print("Please select a valid choice for histogram_type: 'letter_levels' or 'letter_combined' or 'letter_levels_and_combined'")
            print("or 'kineme_levels' or 'kineme_combined' or 'kineme_levels_and_combined'")

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
    def get_data(self, filename):
        header, columns = self.read_csv(filename)
        #print("header: ", header)
        self.frame_no = list(map(int, columns["frame_no"]))
        self.letter = list(map(str, columns["letter"]))
        self.magnitude = list(map(float, columns["magnitude"]))
        self.level = list(map(float, columns["level"]))
        self.size = list(map(float, columns["size"]))
        self.face_no = list(map(int, list(map(float, columns["face_no"]))))
        self.speaking = list(map(int, list(map(float, columns["speaking"]))))
        self.conf_part = list(map(int, list(map(float, columns["conf_part"]))))
        self.fps = list(map(float, columns["fps"]))

        return self.frame_no, self.letter, self.magnitude, self.level, self.size, self.face_no, self.speaking, self.conf_part, self.fps

    def save_histograms(self, HistFolder, filenameBase, filenameEndLevels, filenameEndCombined, filenameEndBoth, histograms, selected_levels, letter_dictionary, segment_size, segment_id, histogram_type = 'letter_levels_and_combined'):
        #save histpragam parts to csv file
        np.set_printoptions(precision=4)
        if not os.path.exists(HistFolder):
            os.makedirs(HistFolder)

        letters = f"{letter_dictionary}"
        names = ['levels', f"{letters}", "segment_id", "segment_size"]
        #print("names: ", names)

        if histogram_type == 'letter_levels' or histogram_type == 'kineme_levels':
            open_file = f'{HistFolder}{filenameBase}_{filenameEndLevels}_{segment_id}.csv'
            with open(open_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(names)
                for idl in range(len(selected_levels)):
                    histogram = histograms[idl]
                    segment_id = idl
                    writer.writerow([selected_levels[idl], np.round(histogram, 4).tolist(), segment_id, segment_size])

        elif histogram_type == 'letter_combined' or histogram_type == 'kineme_combined':
            open_file = f'{HistFolder}{filenameBase}_{filenameEndCombined}_{segment_id}.csv'
            with open(open_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(names)
                histogram = np.array(histograms)
                segment_id = idp
                writer.writerow(['all', np.round(histogram, 4).tolist(), segment_id, segment_size])
        elif histogram_type == 'letter_levels_and_combined' or histogram_type == 'kineme_levels_and_combined':
            open_file = f'{HistFolder}{filenameBase}_{filenameEndBoth}_{segment_id}.csv'
            with open(open_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(names)
                for idl in range(len(selected_levels)):
                    histogram = np.array(histograms[idl])
                    segment_id = idl
                    writer.writerow([selected_levels[idl], np.round(histogram, 4).tolist(), segment_id, segment_size])
                writer.writerow(['all', np.round(histograms[len(selected_levels)], 4).tolist(), segment_id, segment_size])
        else:
            print("Please select a valid choice for histogram_type: 'letter_levels' or 'letter_combined' or 'letter_levels_and_combined'")
            print("or 'kineme_levels' or 'kineme_combined' or 'kineme_levels_and_combined'")


    def select_histogram(self, histogram_nonseparated, histogram_speaker, histogram_listener):

        speaking_listening = self.hist_opts['speaking_listening']
        if speaking_listening == 'speaking':
            histogram = histogram_speaker
            histogram = [item for array in histogram for sublist in array for item in sublist]
        elif speaking_listening == 'listening':
            histogram =histogram_listener
            histogram = [item for array in histogram for sublist in array for item in sublist]
        elif speaking_listening == 'nonseparated' :
            histogram =histogram_nonseparated
            histogram = [item for array in histogram for sublist in array for item in sublist]
        elif speaking_listening == 'separated_and_combined':
            flattened_list1 = [item for array in histogram_speaker for sublist in array for item in sublist]
            flattened_list2 = [item for array in histogram_listener for sublist in array for item in sublist]
            histogram = sum([flattened_list1, flattened_list2], [])
        else:
            print("Please select a valid choice for speaking_listening: 'speaking' or 'listening' or 'nonseparated' or 'separated_and_combined'")
        #flattened_list = [item for array in histogram for sublist in array for item in sublist]
        return histogram
        
    def generate_histograms(self,all_letters):

        if self.hist_opts['histogram_segment_type'] == 'segment_size':
            segment_size = self.hist_opts['histogram_segment_size'] * self.hist_opts['fps']
            no_of_segments = self.get_number_of_segments(len(all_letters['letter']), segment_size)
            if segment_size < 0:  # -1 for all frames
                segment_size = len(all_letters['letter'])
                no_of_segments = 1
        elif self.hist_opts['histogram_segment_type'] == 'segment_number':
            segment_size = len(all_letters['letter']) // self.hist_opts['histogram_segment_number']
            no_of_segments = self.hist_opts['histogram_segment_number']
        else:
            print(" Histogram segment type is not valid", self.hist_opts['histogram_segment_type'])
            print("Please select a valid choice for histogram_segment_type: 'segment_size' or 'segment_number'")
            return None

        letter = all_letters['letter']
        magnitude = all_letters['magnitude']
        level = all_letters['level']
        fps = self.hist_opts['fps']
        selected_levels = self.hist_opts['selected_levels']
        letter_dictionary = self.hist_opts['letter_dictionary']

        hist_value = self.hist_opts['hist_value']
        video_length = len(letter)
        kineme_type = self.hist_opts['kineme_type']
        #print("segment_size: ", segment_size, " fps: ", fps)
        #no_of_segments = self.get_number_of_segments(video_length, segment_size)
        print("length : ",len(letter), "no_of_segments: ", no_of_segments, " segment_size: ", segment_size)
        selected_letters = self.kineme_types_mapping.get(kineme_type)

        histogram_all_parts_levels = []
        histogram_all_parts_combined = []
        histogram_all_parts_both = []
        #print("selected levels              : ", selected_levels)
        #print("kineme_type                  : ", kineme_type)
        #print("selected letters or kinemes  :", selected_letters)
        #print("hist_value                   : ", hist_value)
        #print("histogram_type               : ", self.hist_opts['histogram_type'])
        #print("speaking_listening choice    : ", self.hist_opts['speaking_listening'])

        for ids in range(no_of_segments):
            letters_part = letter[ids * segment_size: ids * segment_size + segment_size]
            levels_part = level[ids * segment_size: ids * segment_size + segment_size]
            magnitudes_part = magnitude[ids * segment_size: ids * segment_size + segment_size]
            histograms = self.get_level_histograms(letters_part, levels_part, magnitudes_part, selected_levels,
                                                   selected_letters, hist_value)
            histogram1D = self.get_histogram1D(histograms, letters_part, levels_part, magnitudes_part, selected_levels,
                                               selected_letters)
            histogram_combined = self.combine_histograms(histograms, histogram1D, selected_levels, selected_letters)
            if self.hist_opts['histogram_type'] == 'letter_levels' or self.hist_opts['histogram_type'] == 'kineme_levels':
                #histograms = self.get_level_histograms(letters_part, levels_part, magnitudes_part, selected_levels, selected_letters, hist_value)
                if self.hist_opts['show_histogram_plots'] or self.hist_opts['save_histogram_plots']:
                    self.plot_histograms(self.kwargs['plotFolder'], self.kwargs['filename_base'], "Levels", "Combined",
                                     "Both", histograms, selected_levels, selected_letters, ids,
                                     f"{self.hist_opts['histogram_type']}")
                if self.hist_opts['save_histograms_to_file']:
                    self.save_histograms(self.kwargs['histFolder'], self.kwargs['filename_base'], "Levels", "Combined",
                                     "Both", histograms, selected_levels, selected_letters, segment_size, ids,
                                     f"{self.hist_opts['histogram_type']}")
                histogram_all_parts_levels.append(histograms.copy())
            elif self.hist_opts['histogram_type'] == 'letter_combined' or self.hist_opts['histogram_type'] == 'kineme_combined':
                #histogram1D = self.get_histogram1D(histograms, letters_part, levels_part, magnitudes_part, selected_levels, selected_letters)
                if self.hist_opts['show_histogram_plots'] or self.hist_opts['save_histogram_plots']:
                    self.plot_histograms(self.kwargs['plotFolder'], self.kwargs['filename_base'], "Levels", "Combined",
                                     "Both", histogram1D, selected_levels, selected_letters, ids,
                                     f"{self.hist_opts['histogram_type']}")
                if self.hist_opts['save_histograms_to_file']:
                    self.save_histograms(self.kwargs['histFolder'], self.kwargs['filename_base'], "Levels", "Combined",
                                     "Both", histogram1D, selected_levels, selected_letters, segment_size, ids,
                                     f"{self.hist_opts['histogram_type']}")
                histogram_all_parts_combined.append([histogram1D.copy()])
            elif self.hist_opts['histogram_type'] == 'letter_levels_and_combined' or self.hist_opts['histogram_type'] == 'kineme_levels_and_combined':
                #histogram_combined = self.combine_histograms(histograms, histogram1D, selected_levels, selected_letters)
                if self.hist_opts['show_histogram_plots'] or self.hist_opts['save_histogram_plots']:
                    self.plot_histograms(self.kwargs['plotFolder'], self.kwargs['filename_base'], "Levels", "Combined",
                                     "Both", histogram_combined, selected_levels, selected_letters, ids,
                                     f"{self.hist_opts['histogram_type']}")
                if self.hist_opts['save_histograms_to_file']:
                    self.save_histograms(self.kwargs['histFolder'], self.kwargs['filename_base'], "Levels", "Combined",
                                     "Both", histogram_combined, selected_levels, selected_letters, segment_size, ids,
                                     f"{self.hist_opts['histogram_type']}")
                histogram_all_parts_both.append(histogram_combined.copy())
        if self.hist_opts['histogram_type'] == 'letter_levels' or self.hist_opts['histogram_type'] == 'kineme_levels':
            return histogram_all_parts_levels
        elif self.hist_opts['histogram_type'] == 'letter_combined' or self.hist_opts['histogram_type'] == 'kineme_combined':
            return histogram_all_parts_combined
        elif self.hist_opts['histogram_type'] == 'letter_levels_and_combined' or self.hist_opts['histogram_type'] == 'kineme_levels_and_combined':
            return histogram_all_parts_both
        else:
            print("Please select a valid choice for histogram_type: ")
            print(""'letter_levels' or 'letter_combined' or 'letter_levels_and_combined'"")
            print("'or kineme_levels' or 'kineme_combined' or 'kineme_levels_and_combined'")
            return None


