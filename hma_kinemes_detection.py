import csv, os
import numpy as np
import matplotlib.pyplot as plt
#from pipeline8_letter_histograms import LetterHistograms
import sys

#Reads letter data_in from csv file
#returns kineme groups detected at each level (scale) such as one-nod, two-nod, three-nod, singleton etc.

class Kinemes_Class():
    def __init__(self, kineme_opts):
        self.kineme_opts = kineme_opts
        self.kineme_type = kineme_opts['kineme_type']
        self.kineme_types_mapping = {
        'single_letters': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's','t', 'u', 'v', 'w', 'x', 'y', 'z', '_'],
        'singleton' : ['i', 'r', 'u', 'x', 'y', 'z'],
        'one_sweep' : ['u', 'x'],
        'two_sweep' : ['uu', 'ux', 'xu', 'xx'],
        'three_sweep' : ['uuu', 'uux', 'uxu', 'uxx', 'xuu', 'xux', 'xxu', 'xxx'],
        'one_nod' : ['i', 'r'],
        'two_nod' : ['ir', 'ri', 'ii', 'rr'],
        'three_nod' : ['iii', 'iir', 'iri', 'irr', 'rii', 'rir', 'rri', 'rrr'],
        }
        #self.selected_kineme = self.kineme_types_mapping.get[self.kineme_type]
        self.kinemeObject = self.kineme(0)
    class kineme():
        def __init__(self, position):
            self.position = position
            self.frame = 0
            self.level = 0
            self.size = round(0.0, 4)
            self.name = ''
            self.type = 'one_nod'  # or 'singleton' or 'two_nod' or 'three_nod' or 'one_sweep' or 'two_sweep' or 'three_sweep'
            self.angle = round(0.0, 4)
            self.magnitude = round(0.0, 4)
            self.first_frame = self.position - int(self.size)
            self.last_frame = self.position + int(self.size)
    def decompose_to_levels(self, letter, magnitude, level, size, selected_levels):
        decomposed_frames = [[] for _ in range(len(selected_levels))]
        decomposed_letters = [[] for _ in range(len(selected_levels))]
        decomposed_magnitudes = [[] for _ in range(len(selected_levels))]
        decomposed_sizes = [[] for _ in range(len(selected_levels))]
        decomposed_levels = [[] for _ in range(len(selected_levels))]
        for ln in range(len(letter)):
            for idl in range(len(selected_levels)):
                if level[ln] == selected_levels[idl]:
                    decomposed_frames[idl].append(ln)
                    decomposed_letters[idl].append(letter[ln])
                    decomposed_magnitudes[idl].append(magnitude[ln])
                    decomposed_sizes[idl].append(size[ln])
                    decomposed_levels[idl].append(level[ln])
                if letter[ln] == "" or letter[ln] == "-":
                        #magnitude, size and levels are set to 0 for "" "-"" and " " letters.
                        # if levels[id] is zero it means the letter is "" or "-".
                        decomposed_letters[idl].append(letter[ln])
                        decomposed_frames[idl].append(ln)
                        decomposed_magnitudes[idl].append(0.0)
                        decomposed_sizes[idl].append(0.0)
                        decomposed_levels[idl].append(0.0)
        return decomposed_frames, decomposed_letters, decomposed_magnitudes,  decomposed_levels, decomposed_sizes
    def squeeze_letter(self, decomposed_frames, decomposed_letters, decomposed_magnitudes, decomposed_levels, decomposed_sizes, selected_levels):
        squeezed_frames = [[] for _ in range(len(selected_levels))]
        squeezed_letters= [[] for _ in range(len(selected_levels))]
        squeezed_magnitudes = [[] for _ in range(len(selected_levels))]
        squeezed_levels = [[] for _ in range(len(selected_levels))]
        squeezed_sizes = [[] for _ in range(len(selected_levels))]
        for idl in range(len(selected_levels)):
            for ind in range(len(decomposed_letters[idl])):
                if decomposed_letters[idl][ind] == '' or decomposed_letters[idl][ind] == ' ':
                    pass
                else:
                    #print(decomposed_letters[idl][ind] , end="")
                    squeezed_frames[idl].append(decomposed_frames[idl][ind])
                    squeezed_letters[idl].append(decomposed_letters[idl][ind])
                    squeezed_magnitudes[idl].append(decomposed_magnitudes[idl][ind])
                    squeezed_levels[idl].append(decomposed_levels[idl][ind])
                    squeezed_sizes[idl].append(decomposed_sizes[idl][ind])
        return squeezed_frames, squeezed_letters, squeezed_magnitudes, squeezed_levels, squeezed_sizes

    def detect_single_letters(self, decomposedSqueezed_frames, decomposedSqueezed_letters, decomposedSqueezed_magnitudes, decomposedSqueezed_levels, decomposedSqueezed_sizes, selected_levels, single_letters):
        detected_single_letters = [ [[] for _ in range(len(selected_levels))  ] for _ in  range(len(single_letters))]

        #print("single_letters", single_letters)
        for idl in range(len(selected_levels)):
            for ind in range(len(decomposedSqueezed_letters[idl])):
                for ond in range(len(single_letters)):
                    #print("idl, ind decomposedSqueezed_letters[idl][ind]", idl, ind,
                          #decomposedSqueezed_letters[idl][ind], end="")
                    if decomposedSqueezed_letters[idl][ind] == single_letters[ond]:
                        found_kineme = self.kineme(ind)
                        found_kineme.type = 'single_letter'
                        found_kineme.name = single_letters[ond]
                        found_kineme.position = decomposedSqueezed_frames[idl][ind]
                        found_kineme.level = decomposedSqueezed_levels[idl][ind]
                        found_kineme.size = decomposedSqueezed_sizes[idl][ind]
                        found_kineme.magnitude = round(decomposedSqueezed_magnitudes[idl][ind],4)
                        found_kineme.first_frame = found_kineme.position - int(found_kineme.size)
                        found_kineme.last_frame = found_kineme.position + int(found_kineme.size)
                        detected_single_letters[ond][idl].append(found_kineme)
        return detected_single_letters


    def detect_one_nod(self, decomposedSqueezed_frames, decomposedSqueezed_letters, decomposedSqueezed_magnitudes, decomposedSqueezed_levels, decomposedSqueezed_sizes, selected_levels, one_nod):
        detected_one_nods = [ [[] for _ in range(len(selected_levels))  ] for _ in  range(len(one_nod))]

        for idl in range(len(selected_levels)):
            for ind in range(len(decomposedSqueezed_letters[idl])):
                for ond in range(len(one_nod)):
                    if decomposedSqueezed_letters[idl][ind] == one_nod[ond]:
                        found_kineme = self.kineme(ind)
                        found_kineme.type = 'one_nod'
                        found_kineme.name = one_nod[ond]
                        found_kineme.position = decomposedSqueezed_frames[idl][ind]
                        found_kineme.level = decomposedSqueezed_levels[idl][ind]
                        found_kineme.size = decomposedSqueezed_sizes[idl][ind]
                        found_kineme.magnitude = round(decomposedSqueezed_magnitudes[idl][ind],4)
                        found_kineme.first_frame = found_kineme.position - int(found_kineme.size)
                        found_kineme.last_frame = found_kineme.position + int(found_kineme.size)
                        detected_one_nods[ond][idl].append(found_kineme)
        return detected_one_nods

    def detect_two_nod(self, decomposedSqueezed_frames, decomposedSqueezed_letters, decomposedSqueezed_magnitudes, decomposedSqueezed_levels,
                       decomposedSqueezed_sizes, selected_levels, two_nod):
        detected_two_nods = [[[] for _ in range(len(selected_levels))] for _ in range(len(two_nod))]

        for idl in range(len(selected_levels)):
            for ind in range(1,len(decomposedSqueezed_letters[idl])):
                first_letter = decomposedSqueezed_letters[idl][ind-1]
                second_letter = decomposedSqueezed_letters[idl][ind]
                for ond in range(len(two_nod)):
                    if first_letter == two_nod[ond][0] and second_letter == two_nod[ond][1]:
                        first_kineme = self.kineme(ind-1)
                        first_kineme.type = 'one_nod'
                        first_kineme.name = two_nod[ond]
                        first_kineme.position = decomposedSqueezed_frames[idl][ind-1]
                        first_kineme.level = idl
                        first_kineme.size = decomposedSqueezed_sizes[idl][ind-1]
                        first_kineme.magnitude = decomposedSqueezed_magnitudes[idl][ind-1]
                        first_kineme.first_frame = int(first_kineme.position - int(first_kineme.size ))
                        first_kineme.last_frame = int(first_kineme.position + int(first_kineme.size ))
                        second_kineme = self.kineme(ind)
                        second_kineme.type = 'two_nod'
                        second_kineme.name = two_nod[ond]
                        second_kineme.position = decomposedSqueezed_frames[idl][ind]
                        second_kineme.level = idl
                        second_kineme.size = decomposedSqueezed_sizes[idl][ind]
                        second_kineme.magnitude = decomposedSqueezed_magnitudes[idl][ind]
                        second_kineme.first_frame = second_kineme.position - int(second_kineme.size )
                        second_kineme.last_frame = second_kineme.position + int(second_kineme.size )
                        two_nod_kineme = self.kineme(ind)
                        two_nod_kineme.type = 'two_nod'
                        two_nod_kineme.name = two_nod[ond]
                        two_nod_kineme.position = decomposedSqueezed_frames[idl][ind]
                        two_nod_kineme.level = decomposedSqueezed_levels[idl][ind]
                        two_nod_kineme.size = round((decomposedSqueezed_sizes[idl][ind-1] + decomposedSqueezed_sizes[idl][ind]) / 2, 2)
                        two_nod_kineme.magnitude = round((decomposedSqueezed_magnitudes[idl][ind-1] + decomposedSqueezed_magnitudes[idl][ind])/2, 4)
                        two_nod_kineme.first_frame = first_kineme.first_frame
                        two_nod_kineme.last_frame = second_kineme.last_frame
                        detected_two_nods[ond][idl].append(two_nod_kineme)
        return detected_two_nods

    def detect_three_nod(self, decomposedSqueezed_frames, decomposedSqueezed_letters, decomposedSqueezed_magnitudes, decomposedSqueezed_levels,
                       decomposedSqueezed_sizes, selected_levels, three_nod):
        detected_three_nods = [[[] for _ in range(len(selected_levels))] for _ in range(len(three_nod))]

        for idl in range(len(selected_levels)):
            for ind in range(1,len(decomposedSqueezed_letters[idl]) -1 ):
                first_letter = decomposedSqueezed_letters[idl][ind-1]
                second_letter = decomposedSqueezed_letters[idl][ind]
                third_letter = decomposedSqueezed_letters[idl][ind + 1]
                for ond in range(len(three_nod)):
                    if first_letter == three_nod[ond][0] and second_letter == three_nod[ond][1] and third_letter == three_nod[ond][2]:
                        first_kineme = self.kineme(ind-1)
                        first_kineme.type = 'one_nod'
                        first_kineme.name = three_nod[ond]
                        first_kineme.position = decomposedSqueezed_frames[idl][ind-1]
                        first_kineme.level = idl
                        first_kineme.size = decomposedSqueezed_sizes[idl][ind-1]
                        first_kineme.magnitude = decomposedSqueezed_magnitudes[idl][ind-1]
                        first_kineme.first_frame = first_kineme.position - int(first_kineme.size )
                        first_kineme.last_frame = first_kineme.position + int(first_kineme.size )
                        second_kineme = self.kineme(ind)
                        second_kineme.type = 'three_nod'
                        second_kineme.name = three_nod[ond]
                        second_kineme.position = decomposedSqueezed_frames[idl][ind]
                        second_kineme.level = idl
                        second_kineme.size = decomposedSqueezed_sizes[idl][ind]
                        second_kineme.magnitude = decomposedSqueezed_magnitudes[idl][ind]
                        second_kineme.first_frame = second_kineme.position - int(second_kineme.size )
                        second_kineme.last_frame = second_kineme.position + int(second_kineme.size )
                        third_kineme = self.kineme(ind + 1)
                        third_kineme.type = 'three_nod'
                        third_kineme.name = three_nod[ond]
                        third_kineme.position = decomposedSqueezed_frames[idl][ind+1]
                        third_kineme.level = idl
                        third_kineme.size = decomposedSqueezed_sizes[idl][ind + 1]
                        third_kineme.magnitude = decomposedSqueezed_magnitudes[idl][ind + 1]
                        third_kineme.first_frame = third_kineme.position - int(third_kineme.size )
                        third_kineme.last_frame = third_kineme.position + int(third_kineme.size )

                        three_nod_kineme = self.kineme(ind)
                        three_nod_kineme.type = 'three_nod'
                        three_nod_kineme.name = three_nod[ond]
                        three_nod_kineme.position = decomposedSqueezed_frames[idl][ind]
                        three_nod_kineme.level = decomposedSqueezed_levels[idl][ind]
                        three_nod_kineme.size = round((decomposedSqueezed_sizes[idl][ind-1] + decomposedSqueezed_sizes[idl][ind] + decomposedSqueezed_sizes[idl][ind + 1]) / 3, 2)
                        three_nod_kineme.magnitude = round((decomposedSqueezed_magnitudes[idl][ind-1] + decomposedSqueezed_magnitudes[idl][ind-1] + decomposedSqueezed_magnitudes[idl][ind+1]) / 3., 4)
                        three_nod_kineme.first_frame = first_kineme.first_frame
                        three_nod_kineme.last_frame = third_kineme.last_frame
                        detected_three_nods[ond][idl].append(three_nod_kineme)
        return detected_three_nods

    def detect_singleton(self, decomposedSqueezed_frames, decomposedSqueezed_letters, decomposedSqueezed_magnitudes, decomposedSqueezed_levels, decomposedSqueezed_sizes, selected_levels, singleton):
        detected_singletons = [ [[] for _ in range(len(selected_levels))  ] for _ in  range(len(singleton))]
        for idl in range(len(selected_levels)):
            for ind in range(len(decomposedSqueezed_letters[idl])):
                for ond in range(len(singleton)):
                    if decomposedSqueezed_letters[idl][ind] == singleton[ond]:
                        found_kineme = self.kineme(ind)
                        found_kineme.type = 'singleton'
                        found_kineme.name = singleton[ond]
                        found_kineme.position = decomposedSqueezed_frames[idl][ind]
                        found_kineme.level = decomposedSqueezed_levels[idl][ind]
                        found_kineme.size = decomposedSqueezed_sizes[idl][ind]
                        found_kineme.magnitude = round(decomposedSqueezed_magnitudes[idl][ind],4)
                        found_kineme.first_frame = found_kineme.position - int(found_kineme.size )
                        found_kineme.last_frame = found_kineme.position + int(found_kineme.size )
                        detected_singletons[ond][idl].append(found_kineme)
        return detected_singletons

    def detect_one_sweep(self, decomposedSqueezed_frames, decomposedSqueezed_letters, decomposedSqueezed_magnitudes, decomposedSqueezed_levels, decomposedSqueezed_sizes, selected_levels, one_sweep):
        detected_one_sweeps = [ [[] for _ in range(len(selected_levels))  ] for _ in  range(len(one_sweep))]

        for idl in range(len(selected_levels)):
            for ind in range(len(decomposedSqueezed_letters[idl])):
                for ond in range(len(one_sweep)):
                    if decomposedSqueezed_letters[idl][ind] == one_sweep[ond]:
                        found_kineme = self.kineme(ind)
                        found_kineme.type = 'one_sweep'
                        found_kineme.name = one_sweep[ond]
                        found_kineme.position = decomposedSqueezed_frames[idl][ind]
                        found_kineme.level = decomposedSqueezed_levels[idl][ind]
                        found_kineme.size = decomposedSqueezed_sizes[idl][ind]
                        found_kineme.magnitude = round(decomposedSqueezed_magnitudes[idl][ind],4)
                        found_kineme.first_frame = found_kineme.position - int(found_kineme.size )
                        found_kineme.last_frame = found_kineme.position + int(found_kineme.size )
                        detected_one_sweeps[ond][idl].append(found_kineme)
        return detected_one_sweeps

    def detect_two_sweep(self, decomposedSqueezed_frames, decomposedSqueezed_letters, decomposedSqueezed_magnitudes, decomposedSqueezed_levels,
                       decomposedSqueezed_sizes, selected_levels, two_sweep):
        detected_two_sweeps = [[[] for _ in range(len(selected_levels))] for _ in range(len(two_sweep))]

        for idl in range(len(selected_levels)):
            for ind in range(1,len(decomposedSqueezed_letters[idl])):
                first_letter = decomposedSqueezed_letters[idl][ind-1]
                second_letter = decomposedSqueezed_letters[idl][ind]
                for ond in range(len(two_sweep)):
                    if first_letter == two_sweep[ond][0] and second_letter == two_sweep[ond][1]:
                        first_kineme = self.kineme(ind-1)
                        first_kineme.type = 'one_sweep'
                        first_kineme.name = two_sweep[ond]
                        first_kineme.position = decomposedSqueezed_frames[idl][ind-1]
                        first_kineme.level = idl
                        first_kineme.size = decomposedSqueezed_sizes[idl][ind-1]
                        first_kineme.magnitude = decomposedSqueezed_magnitudes[idl][ind-1]
                        first_kineme.first_frame = first_kineme.position - int(first_kineme.size )
                        first_kineme.last_frame = first_kineme.position + int(first_kineme.size )
                        second_kineme = self.kineme(ind)
                        second_kineme.type = 'two_sweep'
                        second_kineme.name = two_sweep[ond]
                        second_kineme.position = decomposedSqueezed_frames[idl][ind]
                        second_kineme.level = idl
                        second_kineme.size = decomposedSqueezed_sizes[idl][ind]
                        second_kineme.magnitude = decomposedSqueezed_magnitudes[idl][ind]
                        second_kineme.first_frame = second_kineme.position - int(second_kineme.size )
                        second_kineme.last_frame = second_kineme.position + int(second_kineme.size )
                        two_sweep_kineme = self.kineme(ind)
                        two_sweep_kineme.type = 'two_sweep'
                        two_sweep_kineme.name = two_sweep[ond]
                        two_sweep_kineme.position = decomposedSqueezed_frames[idl][ind]
                        two_sweep_kineme.level = decomposedSqueezed_levels[idl][ind]
                        two_sweep_kineme.size = round((decomposedSqueezed_sizes[idl][ind-1] + decomposedSqueezed_sizes[idl][ind]) / 2, 2)
                        two_sweep_kineme.magnitude = round((decomposedSqueezed_magnitudes[idl][ind-1] + decomposedSqueezed_magnitudes[idl][ind])/2, 4)
                        two_sweep_kineme.first_frame = first_kineme.first_frame
                        two_sweep_kineme.last_frame = second_kineme.last_frame
                        detected_two_sweeps[ond][idl].append(two_sweep_kineme)
        return detected_two_sweeps

    def detect_three_sweep(self, decomposedSqueezed_frames, decomposedSqueezed_letters, decomposedSqueezed_magnitudes, decomposedSqueezed_levels,
                       decomposedSqueezed_sizes, selected_levels, three_sweep):
        detected_three_sweeps = [[[] for _ in range(len(selected_levels))] for _ in range(len(three_sweep))]

        for idl in range(len(selected_levels)):
            for ind in range(1,len(decomposedSqueezed_letters[idl]) -1 ):
                first_letter = decomposedSqueezed_letters[idl][ind-1]
                second_letter = decomposedSqueezed_letters[idl][ind]
                third_letter = decomposedSqueezed_letters[idl][ind + 1]
                for ond in range(len(three_sweep)):
                    if first_letter == three_sweep[ond][0] and second_letter == three_sweep[ond][1] and third_letter == three_sweep[ond][2]:
                        first_kineme = self.kineme(ind-1)
                        first_kineme.type = 'one_sweep'
                        first_kineme.name = three_sweep[ond]
                        first_kineme.position = decomposedSqueezed_frames[idl][ind-1]
                        first_kineme.level = idl
                        first_kineme.size = decomposedSqueezed_sizes[idl][ind-1]
                        first_kineme.magnitude = decomposedSqueezed_magnitudes[idl][ind-1]
                        first_kineme.first_frame = first_kineme.position - int(first_kineme.size )
                        first_kineme.last_frame = first_kineme.position + int(first_kineme.size )
                        second_kineme = self.kineme(ind)
                        second_kineme.type = 'three_sweep'
                        second_kineme.name = three_sweep[ond]
                        second_kineme.position = decomposedSqueezed_frames[idl][ind]
                        second_kineme.level = idl
                        second_kineme.size = decomposedSqueezed_sizes[idl][ind]
                        second_kineme.magnitude = decomposedSqueezed_magnitudes[idl][ind]
                        second_kineme.first_frame = second_kineme.position - int(second_kineme.size )
                        second_kineme.last_frame = second_kineme.position + int(second_kineme.size )
                        third_kineme = self.kineme(ind + 1)
                        third_kineme.type = 'three_sweep'
                        third_kineme.name = three_sweep[ond]
                        third_kineme.position = decomposedSqueezed_frames[idl][ind+1]
                        third_kineme.level = idl
                        third_kineme.size = decomposedSqueezed_sizes[idl][ind + 1]
                        third_kineme.magnitude = decomposedSqueezed_magnitudes[idl][ind + 1]
                        third_kineme.first_frame = third_kineme.position - int(third_kineme.size )
                        third_kineme.last_frame = third_kineme.position + int(third_kineme.size )

                        three_sweep_kineme = self.kineme(ind)
                        three_sweep_kineme.type = 'three_sweep'
                        three_sweep_kineme.name = three_sweep[ond]
                        three_sweep_kineme.position = decomposedSqueezed_frames[idl][ind]
                        three_sweep_kineme.level = decomposedSqueezed_levels[idl][ind]
                        three_sweep_kineme.size = round((decomposedSqueezed_sizes[idl][ind-1] + decomposedSqueezed_sizes[idl][ind] + decomposedSqueezed_sizes[idl][ind + 1]) / 3, 2)
                        three_sweep_kineme.magnitude = round((decomposedSqueezed_magnitudes[idl][ind-1] +  + decomposedSqueezed_magnitudes[idl][ind+1]) / 3., 4)
                        three_sweep_kineme.first_frame = first_kineme.first_frame
                        three_sweep_kineme.last_frame = third_kineme.last_frame
                        detected_three_sweeps[ond][idl].append(three_sweep_kineme)
        return detected_three_sweeps
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

    def get_kinemes(self, all_letters):

        frame_no = all_letters['frame_no']
        letter = all_letters['letter']
        magnitude = all_letters['magnitude']
        level = all_letters['level']
        size =  all_letters['size']
        video_length = len(letter)
        selected_levels = self.kineme_opts['selected_levels']
        #video_length = self.kineme_opts['video_length']
        kineme_type = self.kineme_opts['kineme_type']
        selected_kinemes = self.kineme_types_mapping.get(kineme_type)
        #print("selected_kinemes: (get_kinemes)", selected_kinemes)
        #print("video_length: (get_kinemes)", video_length)

        decomposed_frames, decomposed_letters, decomposed_magnitudes, decomposed_levels, decomposed_sizes = self.decompose_to_levels(
            letter, magnitude, level, size, selected_levels)

        squeezed_frames, squeezed_letters, squeezed_magnitudes, squeezed_levels, squeezed_sizes = self.squeeze_letter(decomposed_frames, decomposed_letters, decomposed_magnitudes, decomposed_levels, decomposed_sizes,
            selected_levels)
        if self.kineme_opts['kineme_type'] == 'one_nod':
            detected_kinemes = self.detect_one_nod(squeezed_frames, squeezed_letters, squeezed_magnitudes,
                                                  squeezed_levels, squeezed_sizes, selected_levels, self.kineme_types_mapping.get('one_nod'))
        elif self.kineme_opts['kineme_type'] == 'two_nod':
            detected_kinemes = self.detect_two_nod(squeezed_frames, squeezed_letters, squeezed_magnitudes,
                                                  squeezed_levels, squeezed_sizes, selected_levels, self.kineme_types_mapping.get('two_nod'))
        elif self.kineme_opts['kineme_type'] == 'three_nod':
            detected_kinemes = self.detect_three_nod(squeezed_frames, squeezed_letters, squeezed_magnitudes,
                                                      squeezed_levels, squeezed_sizes, selected_levels,
                                                      self.kineme_types_mapping.get('three_nod'))
        elif self.kineme_opts['kineme_type'] == 'singleton':
            detected_kinemes = self.detect_singleton(squeezed_frames, squeezed_letters, squeezed_magnitudes,
                                                      squeezed_levels, squeezed_sizes, selected_levels,
                                                      self.kineme_types_mapping.get('singleton'))
        elif self.kineme_opts['kineme_type'] == 'one_sweep':
            detected_kinemes = self.detect_one_sweep(squeezed_frames, squeezed_letters, squeezed_magnitudes,
                                                      squeezed_levels, squeezed_sizes, selected_levels, self.kineme_types_mapping.get('one_sweep'))
        elif self.kineme_opts['kineme_type'] == 'two_sweep':
            detected_kinemes = self.detect_two_sweep(squeezed_frames, squeezed_letters, squeezed_magnitudes,
                                                      squeezed_levels,squeezed_sizes, selected_levels, self.kineme_types_mapping.get('two_sweep'))
        elif self.kineme_opts['kineme_type'] == 'three_sweep':
            detected_kinemes = self.detect_three_sweep(squeezed_frames, squeezed_letters, squeezed_magnitudes,
                                                          squeezed_levels,
                                                          squeezed_sizes, selected_levels, self.kineme_types_mapping.get('three_sweep'))
        elif self.kineme_opts['kineme_type'] == 'single_letters':
            detected_kinemes = self.detect_single_letters(squeezed_frames, squeezed_letters, squeezed_magnitudes,
                                                      squeezed_levels, squeezed_sizes, selected_levels, self.kineme_types_mapping.get('single_letters'))
        else:
            print("Invalid option")

        kineme_ = {'frame': [0.0] * video_length, 'level': [0] * video_length, 'size': [0.0] * video_length,
                   'letter': [''] * video_length, 'type': [''] * video_length,
                   'angle': [0.0] * video_length, 'magnitude': [0.0] * video_length,
                   'first_frame': [0] * video_length, 'last_frame': [0] * video_length}
        for i in range(video_length):
            kineme_['frame'][i] = i
        for twd in range(len(selected_kinemes)):
            for idl in range(len(self.kineme_opts['selected_levels'])):
                for kineme in detected_kinemes[twd][idl]:
                    frame_no = kineme.position
                    kineme_['frame'][frame_no] = kineme.position
                    kineme_['level'][frame_no] = kineme.level
                    kineme_['size'][frame_no] = kineme.size
                    kineme_['letter'][frame_no] = kineme.name
                    kineme_['type'][frame_no] = kineme.type
                    kineme_['angle'][frame_no] = kineme.angle
                    kineme_['magnitude'][frame_no] = kineme.magnitude
                    kineme_['first_frame'][frame_no] = kineme.first_frame
                    kineme_['last_frame'][frame_no] = kineme.last_frame
        return kineme_
    def main(self):
        from hma_extract_video_clips import  ExtractVideoClips as evc
        letterFolder = "/Users/muhittingokmen/Dropbox/CHOP-MEF-data_in/monodic/letters/"
        kinemeFolder = "/Users/muhittingokmen/Dropbox/CHOP-MEF-data_in/monodic/kinemes/"
        clipFolder = "/Users/muhittingokmen/Dropbox/CHOP-MEF-data_in/monodic/kineme_clips/"
        if not os.path.exists(kinemeFolder):
            os.makedirs(kinemeFolder)
        if not os.path.exists(clipFolder):
            os.makedirs(clipFolder)
        #videoFolder = "/Users/muhittingokmen/Dropbox/CHOP-MEF-data_in/monodic/augmented_videos/"
        videoFolder = "/Users/muhittingokmen/Dropbox/CHOP-MEF-data_in/monodic/videos/"

        letterFiles = sorted(os.listdir(letterFolder))
        letterFiles = letterFiles[0:2]
        #print("letterFiles: ", letterFiles)
        # sys.exit()
        for letterFile in sorted(letterFiles):
            if not letterFile.endswith(".csv"):
                continue
            file_path = f"{letterFolder}{letterFile}"
            filenameBase = letterFile.replace('_letters.csv', '')
            filenameBase = filenameBase.replace('_augmented.csv', '')
            #filenameBase = filenameBase + '_kinemes'
            filenameEndOneNod = '_one_nod'
            filenameEndTwoNod = '_two_nod'
            filenameEndThreeNod = '_three_nod'
            filenameEndSingleton = '_singleton'
            filenameEndOneSweep = '_one_sweep'
            filenameEndTwoSweep = '_two_sweep'
            filenameEndThreeSweep = '_three_sweep'
            filenameEndSingleLetters = '_single_letters'

            #hist = LetterHistograms(file_path)
            kinemesObj = Kinemes_Class(self.kineme_opts)

            kinemesObj.file_path = file_path
            frame_no, letter, magnitude, level, size, face_no, speaking, conf_part, fps = kinemesObj.get_data(file_path)

            selected_levels = [1, 4, 7, 10, 13, 16, 19]

            decomposed_frames, decomposed_letters, decomposed_magnitudes,  decomposed_levels, decomposed_sizes = decomp.decompose_to_levels(letter, magnitude, level, size, selected_levels)

            squeezed_frames, squeezed_letters, squeezed_magnitudes, squeezed_levels, squeezed_sizes = kinemesObj.squeeze_letter(decomposed_frames, decomposed_letters, decomposed_magnitudes, decomposed_levels, decomposed_sizes, selected_levels)

            detected_one_nods = kinemesObj.detect_one_nod(squeezed_frames, squeezed_letters, squeezed_magnitudes, squeezed_levels, squeezed_sizes, selected_levels, kinemesObj.one_nod)
            detected_two_nods = kinemesObj.detect_two_nod(squeezed_frames, squeezed_letters, squeezed_magnitudes, squeezed_levels, squeezed_sizes, selected_levels, kinemesObj.two_nod)
            detected_three_nods = kinemesObj.detect_three_nod(squeezed_frames, squeezed_letters, squeezed_magnitudes, squeezed_levels, squeezed_sizes, selected_levels, kinemesObj.three_nod)

            detected_singletons = kinemesObj.detect_singleton(squeezed_frames, squeezed_letters, squeezed_magnitudes, squeezed_levels, squeezed_sizes, selected_levels, kinemesObj.singleton)

            detected_one_sweeps = kinemesObj.detect_one_sweep(squeezed_frames, squeezed_letters, squeezed_magnitudes, squeezed_levels,
                                                      squeezed_sizes, selected_levels, kinemesObj.one_sweep)
            detected_two_sweeps = kinemesObj.detect_two_sweep(squeezed_frames,squeezed_letters, squeezed_magnitudes, squeezed_levels,
                                                      squeezed_sizes, selected_levels, kinemesObj.two_sweep)
            detected_three_sweeps = kinemesObj.detect_three_sweep(squeezed_frames, squeezed_letters, squeezed_magnitudes, squeezed_levels,
                                                          squeezed_sizes, selected_levels, kinemesObj.three_sweep)
            #video_file = videoFolder + filenameBase + '_augmented.mp4'
            video_file = videoFolder + filenameBase + '.mp4'
            for idl in range(len(selected_levels)):
                #print("single_letters levels:", idl)
                for ond in range(len(kinemesObj.single_letters)):
                    for kineme in detected_single_letters[ond][idl]:
                        #print("kineme single_letters: ", kineme.name, kineme.position, kineme.level, kineme.size,
                              #kineme.magnitude, kineme.type, kineme.first_frame, kineme.last_frame)

                        outputClipFolder = clipFolder + "single_letters/"
                        if not os.path.exists(outputClipFolder):
                            os.makedirs(outputClipFolder)
                        filenameBase = filenameBase.replace('_kinemes', '')
                        output_video_filename = outputClipFolder + filenameBase + filenameEndOneNod + '_level_' + str(
                            kineme.level) + "_" + kineme.name + '.mp4'
                        #print("inputVideoFile: ", video_file)
                        #print("outputVideoFiles: ", output_video_filename)
                        evc.extract_frames(video_file, kineme.first_frame)

            for idl in range(len(selected_levels)):
                #print("one_nod levels:", idl)
                for ond in range(len(kinemesObj.one_nod)):
                    for kineme in detected_one_nods[ond][idl]:
                        #print("kineme one_nod: ", kineme.name, kineme.position, kineme.level, kineme.size, kineme.magnitude, kineme.type, kineme.first_frame, kineme.last_frame)


                        outputClipFolder = clipFolder + "one_nod/"
                        if not os.path.exists(outputClipFolder):
                            os.makedirs(outputClipFolder)
                        filenameBase = filenameBase.replace('_kinemes', '')
                        output_video_filename = outputClipFolder + filenameBase + filenameEndOneNod + '_level_' + str(
                            kineme.level) + "_" + kineme.name + '.mp4'
                        #print("inputVideoFile: ", video_file)
                        #print("outputVideoFiles: ", output_video_filename)
                        evc.extract_frames(video_file, kineme.first_frame, kineme.last_frame, output_video_filename)

            for idl in range(len(selected_levels)):
                #print("two_nod levels:", idl)
                for twd in range(len(kinemesObj.two_nod)):
                    for kineme in detected_two_nods[twd][idl]:

                        outputClipFolder = clipFolder + "two_nod/"
                        if not os.path.exists(outputClipFolder):
                            os.makedirs(outputClipFolder)
                        filenameBase = filenameBase.replace('_kinemes', '')
                        output_video_filename = outputClipFolder + filenameBase + filenameEndTwoNod + '_level_' + str(kineme.level) + "_" + kineme.name + '.mp4'
                        #print("inputVideoFile: ", video_file)
                        #print("outputVideoFiles: ", output_video_filename)
                        evc.extract_frames(video_file, kineme.first_frame, kineme.last_frame, output_video_filename)
                        #print("kineme two_nod: ", kineme.name, kineme.position, kineme.level, kineme.size, kineme.magnitude, kineme.type, kineme.first_frame, kineme.last_frame)

            evc.mosaic_videos(outputClipFolder)
            #sys.exit()

            for idl in range(len(selected_levels)):
                #print("three_nod levels:", idl)
                for twd in range(len(kinemesObj.three_nod)):
                    for kineme in detected_three_nods[twd][idl]:
                        #print("kineme three_nod: ", kineme.name, kineme.position, kineme.level, kineme.size, kineme.magnitude, kineme.type, kineme.first_frame, kineme.last_frame)

                        outputClipFolder = clipFolder + "three_nod/"
                        if not os.path.exists(outputClipFolder):
                            os.makedirs(outputClipFolder)
                        filenameBase = filenameBase.replace('_kinemes', '')
                        output_video_filename = outputClipFolder + filenameBase + filenameEndThreeNod + '_level_' + str(
                            kineme.level) + "_" + kineme.name + '.mp4'
                        #print("inputVideoFile: ", video_file)
                        #print("outputVideoFiles: ", output_video_filename)
                        evc.extract_frames(video_file, kineme.first_frame, kineme.last_frame, output_video_filename)


            for idl in range(len(selected_levels)):
                #print("singleton levels:", idl)
                for ond in range(len(kinemesObj.singleton)):
                    for kineme in detected_singletons[ond][idl]:
                        #print("kineme singleton: ", kineme.name, kineme.position, kineme.level, kineme.size, kineme.magnitude, kineme.type, kineme.first_frame, kineme.last_frame)
                        outputClipFolder = clipFolder + "singleton/"
                        if not os.path.exists(outputClipFolder):
                            os.makedirs(outputClipFolder)
                        filenameBase = filenameBase.replace('_kinemes', '')
                        output_video_filename = outputClipFolder + filenameBase + filenameEndSingleton + '_level_' + str(
                            kineme.level) + "_" + kineme.name + '.mp4'
                        #print("inputVideoFile: ", video_file)
                        #print("outputVideoFiles: ", output_video_filename)
                        evc.extract_frames(video_file, kineme.first_frame, kineme.last_frame, output_video_filename)

            for idl in range(len(selected_levels)):
                #print("one_sweep levels:", idl)
                for ond in range(len(kinemesObj.one_sweep)):
                    for kineme in detected_one_sweeps[ond][idl]:
                        #print("kineme one_sweep: ", kineme.name, kineme.position, kineme.level, kineme.size, kineme.magnitude, kineme.type, kineme.first_frame, kineme.last_frame)
                        outputClipFolder = clipFolder + "one_sweep/"
                        if not os.path.exists(outputClipFolder):
                            os.makedirs(outputClipFolder)
                        filenameBase = filenameBase.replace('_kinemes', '')
                        output_video_filename = outputClipFolder + filenameBase + filenameEndOneSweep + '_level_' + str(
                            kineme.level) + "_" + kineme.name + '.mp4'
                        #print("inputVideoFile: ", video_file)
                        #print("outputVideoFiles: ", output_video_filename)
                        evc.extract_frames(video_file, kineme.first_frame, kineme.last_frame, output_video_filename)

            for idl in range(len(selected_levels)):
                #print("two_sweep levels:", idl)
                for twd in range(len(kinemesObj.two_sweep)):
                    for kineme in detected_two_sweeps[twd][idl]:
                        #print("kineme two_sweep: ", kineme.name, kineme.position, kineme.level, kineme.size, kineme.magnitude, kineme.type, kineme.first_frame, kineme.last_frame)
                        outputClipFolder = clipFolder + "two_sweep/"
                        if not os.path.exists(outputClipFolder):
                            os.makedirs(outputClipFolder)
                        filenameBase = filenameBase.replace('_kinemes', '')
                        output_video_filename = outputClipFolder + filenameBase + filenameEndTwoSweep + '_level_' + str(kineme.level) + "_" + kineme.name + '.mp4'
                        #print("inputVideoFile: ", video_file)
                        #print("outputVideoFiles: ", output_video_filename)
                        evc.extract_frames(video_file, kineme.first_frame, kineme.last_frame, output_video_filename)

            for idl in range(len(selected_levels)):
                #print("three_sweep levels:", idl)
                for twd in range(len(kinemesObj.three_sweep)):
                    for kineme in detected_three_sweeps[twd][idl]:
                        #print("kineme three_sweep: ", kineme.name, kineme.position, kineme.level, kineme.size, kineme.magnitude, kineme.type, kineme.first_frame, kineme.last_frame)
                        outputClipFolder = clipFolder + "three_sweep/"
                        if not os.path.exists(outputClipFolder):
                            os.makedirs(outputClipFolder)
                        filenameBase = filenameBase.replace('_kinemes', '')
                        output_video_filename = outputClipFolder + filenameBase + filenameEndThreeSweep + '_level_' + str(kineme.level) + "_" + kineme.name + '.mp4'
                        #print("inputVideoFile: ", video_file)
                        #print("outputVideoFiles: ", output_video_filename)
                        evc.extract_frames(video_file, kineme.first_frame, kineme.last_frame, output_video_filename)

            #print("Done")

if __name__ == "__main__":
    main()
