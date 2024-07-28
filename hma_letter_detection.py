'''
@author: Muhittin Gokmen
@date: 2024-02-16
'''
import math,sys, time, os, argparse, subprocess, glob, pdb
import argparse, os, math
import csv

import pandas as pd
import numpy as np
import sqlite3
from contextlib import closing


class Letters():
    def __init__(self, letter_opts):
       self.letter_opts = letter_opts

    def get_letters(self, keypoints_yaw, keypoints_pitch, keypoints_roll, x):
        #print("keypoint_yaw[10].pt: ", keypoints_yaw[10].pt)

        self.letter_opts['min_silence'] = int(round(self.letter_opts['min_silence'] * (self.letter_opts['fps']) / 25.))
        self.letter_opts['silent_block_size'] = int(round(self.letter_opts['silent_block_size'] * (self.letter_opts['fps']) / 25.))

        min_silence = int(round(self.letter_opts['min_silence']))
        silent_block_size = int(round(self.letter_opts['silent_block_size']))

        letter_vectors, letter_names, letter_dict = self.get_letter_vectors()
        yaw = x[:, 1].tolist()
        pitch = x[:, 0].tolist()
        roll = x[:, 2].tolist()

        video_frames_max = len(yaw)
        frame_count = len(yaw)
        non_zero_frames = [0] * video_frames_max

        speaking = [self.letter_opts['speaking_id']] * video_frames_max
        conf_part = [self.letter_opts['conf_part_id']] * video_frames_max
        face_no = [self.letter_opts['face_number']] * video_frames_max
        fps = [self.letter_opts['fps']] * video_frames_max
        fps_value = int(round(self.letter_opts['fps']))
        
        all_letters = {'frame_no': [0] * video_frames_max, 'letter': [''] * video_frames_max, 'magnitude': [0.0] * video_frames_max,
                       'letter_yaw_component': [0.0] * video_frames_max, 'letter_pitch_component': [0.0] * video_frames_max,
                       'letter_roll_component': [0.0] * video_frames_max, 'level': [0] * video_frames_max, 'size': [0.0] * video_frames_max,
                       'first_frame': [0] * video_frames_max, 'last_frame': [0] * video_frames_max, 'face_no': [self.letter_opts['face_number']] * video_frames_max,
                       'speaking': [self.letter_opts['speaking_id']] * video_frames_max, 'conf_part': [self.letter_opts['conf_part_id']] * video_frames_max , 'fps': [self.letter_opts['fps']] * video_frames_max}

        for idl in range(video_frames_max):
            all_letters['frame_no'][idl] = idl

        yaw_responses = [0] * video_frames_max
        pitch_responses = [0] * video_frames_max
        roll_responses = [0] * video_frames_max
        pitch_responses_in_yaw_keypoints = [0] * video_frames_max
        roll_responses_in_yaw_keypoints = [0] * video_frames_max
        yaw_responses_in_pitch_keypoints = [0] * video_frames_max
        roll_responses_in_pitch_keypoints = [0] * video_frames_max
        yaw_responses_in_roll_keypoints = [0] * video_frames_max
        pitch_responses_in_roll_keypoints = [0] * video_frames_max

        #print("keypoint shapes: ", len(keypoints_yaw), len(keypoints_pitch), len(keypoints_roll))

        for idk in range(len(keypoints_yaw)):
            frame_no = int(keypoints_yaw[idk].pt)
            video_frame_no = int(keypoints_yaw[idk].pt_video)
            yaw_responses[video_frame_no] = keypoints_yaw[idk].response
            pitch_responses[video_frame_no] = keypoints_yaw[idk].response_pitch
            roll_responses[video_frame_no] = keypoints_yaw[idk].response_roll
            pitch_responses_in_yaw_keypoints[video_frame_no] = keypoints_yaw[idk].response_pitch
            roll_responses_in_yaw_keypoints[video_frame_no] = keypoints_yaw[idk].response_roll

        for idk in range(len(keypoints_pitch)):
            frame_no = int(keypoints_pitch[idk].pt)
            video_frame_no = int(keypoints_pitch[idk].pt_video)
            pitch_responses[video_frame_no] = keypoints_pitch[idk].response
            yaw_responses[video_frame_no] = keypoints_pitch[idk].response_yaw
            roll_responses[video_frame_no] = keypoints_pitch[idk].response_roll
            yaw_responses_in_pitch_keypoints[video_frame_no] = keypoints_pitch[idk].response_yaw
            roll_responses_in_pitch_keypoints[video_frame_no] = keypoints_pitch[idk].response_roll

        for idk in range(len(keypoints_roll)):
            frame_no = int(keypoints_roll[idk].pt)
            video_frame_no = int(keypoints_roll[idk].pt_video)
            roll_responses[video_frame_no] = keypoints_roll[idk].response
            yaw_responses[video_frame_no] = keypoints_roll[idk].response_yaw
            pitch_responses[video_frame_no] = keypoints_roll[idk].response_pitch
            yaw_responses_in_roll_keypoints[video_frame_no] = keypoints_roll[idk].response_yaw
            pitch_responses_in_roll_keypoints[video_frame_no] = keypoints_roll[idk].response_pitch

        # find letters for each frame keypoints by looking the combination of yaw, pitch and roll

        letters = [''] * video_frames_max
        for idk in range(len(keypoints_yaw)):
            frame_no = int(keypoints_yaw[idk].pt)
            video_frame_no = int(keypoints_yaw[idk].pt_video)

            if abs(yaw_responses[video_frame_no]) > abs(pitch_responses[video_frame_no]) and abs(
                    yaw_responses[video_frame_no]) > abs(roll_responses[video_frame_no]):
                letter = self.get_letter_name(pitch_responses_in_yaw_keypoints[video_frame_no], yaw_responses[video_frame_no],
                                    roll_responses_in_yaw_keypoints[video_frame_no], letter_vectors, letter_names)
                letters[video_frame_no] = letter
                all_letters['frame_no'][video_frame_no] = frame_no
                all_letters['letter'][video_frame_no] = letter
                all_letters['magnitude'][video_frame_no] = round(math.sqrt(np.sum(np.square(
                    [yaw_responses[video_frame_no], pitch_responses[video_frame_no], roll_responses[video_frame_no]]))), 4)
                all_letters['letter_yaw_component'][video_frame_no] = yaw_responses[video_frame_no]
                all_letters['letter_pitch_component'][video_frame_no] = pitch_responses[video_frame_no]
                all_letters['letter_roll_component'][video_frame_no] = roll_responses[video_frame_no]
                all_letters['level'][video_frame_no] = int(keypoints_yaw[idk].octave)
                all_letters['size'][video_frame_no] = round(keypoints_yaw[idk].size, 4)
                all_letters['first_frame'][video_frame_no] = int(max(keypoints_yaw[idk].first_frame, 0))
                all_letters['last_frame'][video_frame_no] = int(min(keypoints_yaw[idk].last_frame, frame_count))
                all_letters['speaking'][video_frame_no] = keypoints_yaw[idk].speaking
                all_letters['face_no'][video_frame_no] = keypoints_yaw[idk].face_no
                all_letters['conf_part'][video_frame_no] = keypoints_yaw[idk].conf_part
                all_letters['fps'][video_frame_no] = int(round(keypoints_yaw[idk].fps))
                fps_value = int(round(keypoints_yaw[0].fps))


        for idk in range(len(keypoints_pitch)):
            frame_no = int(keypoints_pitch[idk].pt)
            video_frame_no = int(keypoints_pitch[idk].pt_video)
            if letters[video_frame_no] == '':
                if abs(pitch_responses[video_frame_no]) > abs(yaw_responses[video_frame_no]) and abs(
                        pitch_responses[video_frame_no]) > abs(roll_responses[video_frame_no]):
                    letter = self.get_letter_name(pitch_responses[video_frame_no], yaw_responses_in_pitch_keypoints[video_frame_no],
                                        roll_responses_in_pitch_keypoints[video_frame_no], letter_vectors, letter_names)
                    letters[video_frame_no] = letter
                    all_letters['frame_no'][video_frame_no] = frame_no
                    all_letters['letter'][video_frame_no] = letter
                    all_letters['magnitude'][video_frame_no] = round(math.sqrt(np.sum(np.square(
                        [yaw_responses[video_frame_no], pitch_responses[video_frame_no], roll_responses[video_frame_no]]))), 4)
                    all_letters['letter_yaw_component'][video_frame_no] = yaw_responses[video_frame_no]
                    all_letters['letter_pitch_component'][video_frame_no] = pitch_responses[video_frame_no]
                    all_letters['letter_roll_component'][video_frame_no] = roll_responses[video_frame_no]
                    all_letters['level'][video_frame_no] = int(keypoints_pitch[idk].octave)
                    all_letters['size'][video_frame_no] = round(keypoints_pitch[idk].size, 4)
                    all_letters['first_frame'][video_frame_no] = int(max(keypoints_pitch[idk].first_frame, 0))
                    all_letters['last_frame'][video_frame_no] = int(min(keypoints_pitch[idk].last_frame, frame_count))
                    all_letters['speaking'][video_frame_no] = keypoints_pitch[idk].speaking
                    all_letters['face_no'][video_frame_no] = keypoints_pitch[idk].face_no
                    all_letters['conf_part'][video_frame_no] = keypoints_pitch[idk].conf_part
                    all_letters['fps'][video_frame_no] = int(round(keypoints_pitch[idk].fps))


        for idk in range(len(keypoints_roll)):
            frame_no = int(keypoints_roll[idk].pt)
            video_frame_no = int(keypoints_roll[idk].pt_video)
            if letters[video_frame_no] == '':
                if abs(roll_responses[video_frame_no]) > abs(yaw_responses[video_frame_no]) and abs(
                        roll_responses[video_frame_no]) > abs(pitch_responses[video_frame_no]):
                    letter = self.get_letter_name(pitch_responses_in_roll_keypoints[video_frame_no],
                                                  yaw_responses_in_roll_keypoints[video_frame_no],
                                                  roll_responses[video_frame_no], letter_vectors, letter_names)
                    letters[video_frame_no] = letter
                    all_letters['frame_no'][video_frame_no] = frame_no
                    all_letters['letter'][video_frame_no] = letter
                    all_letters['magnitude'][video_frame_no] = round(math.sqrt(np.sum(np.square(
                        [yaw_responses[video_frame_no], pitch_responses[video_frame_no], roll_responses[video_frame_no]]))), 4)
                    all_letters['letter_yaw_component'][video_frame_no] = yaw_responses[video_frame_no]
                    all_letters['letter_pitch_component'][video_frame_no] = pitch_responses[video_frame_no]
                    all_letters['letter_roll_component'][video_frame_no] = roll_responses[video_frame_no]
                    all_letters['level'][video_frame_no] = int(keypoints_roll[idk].octave)
                    all_letters['size'][video_frame_no] = round(keypoints_roll[idk].size, 4)
                    all_letters['first_frame'][video_frame_no] = int(max(keypoints_roll[idk].first_frame, 0))
                    all_letters['last_frame'][video_frame_no] = int(min(keypoints_roll[idk].last_frame, frame_count))
                    all_letters['speaking'][video_frame_no] = keypoints_roll[idk].speaking
                    all_letters['face_no'][video_frame_no] = keypoints_roll[idk].face_no
                    all_letters['conf_part'][video_frame_no] = keypoints_roll[idk].conf_part
                    all_letters['fps'][video_frame_no] = int(round(keypoints_roll[idk].fps))


        # determine silent frames between letters
        for idl in range(len(letters)):
            if letters[idl] == '' or letters[idl] == ' ' or letters[idl] == '_':
                non_zero_frames[idl] = 0
            else:
                non_zero_frames[idl] = 1

                for k in range(- int(all_letters['size'][idl] / 2), int(all_letters['size'][idl] / 2) + 1):
                    ind = idl + k
                    if ind >= 0 and ind < len(letters):
                        non_zero_frames[ind] = 1

        all_letters['fps'] = [fps_value] * video_frames_max

        letters_out = self.no_motion_letters(non_zero_frames, letters, min_silence, silent_block_size)
        all_letters['letter'] = letters_out
        return all_letters

############################################################################################################
    def get_speaker_listener_letters(self, all_letters, s):

        # Convert the dictionary to a DataFrame
        all_letters_df = pd.DataFrame(all_letters)
        #convert ss to numpy array
        #s = pd.DataFrame(s_dict)
        #s = np.array(s) if isinstance(s, list) else s

        # Create listener_letters if s != 0, then replace all values with -1
        listener_letters_df = all_letters_df.copy()
        listener_letters_df.loc[s != 0, :] = -1
        # preserve '-' letters in speaker parts
        listener_letters_df['letter'] = np.where((all_letters_df['letter'] == '-') & (s == 1), '-', '')
        # recover the original letters in the listener part
        listener_letters_df['letter'] = np.where(s == 0, all_letters_df['letter'], ' ')

        # Create speaker_letters if s != 1, then replace all values with -1
        speaker_letters_df = all_letters_df.copy()
        speaker_letters_df.loc[s != 1, :] = -1
        # preserve '-' letters in speaker parts
        speaker_letters_df['letter'] = np.where((all_letters_df['letter'] == '-') & (s == 0), '-', '')
        # recover the original letters in the speaker part
        speaker_letters_df['letter'] = np.where(s == 1, all_letters_df['letter'], speaker_letters_df['letter'])

        #update the 'speaking' column
        all_letters_df['speaking'] = s
        listener_letters_df['speaking'] = s
        speaker_letters_df['speaking'] = s



        #convert DataFrames to dictionaries
        all_letters = all_letters_df.to_dict(orient='list')
        listener_letters = listener_letters_df.to_dict(orient='list')
        speaker_letters = speaker_letters_df.to_dict(orient='list')
        self.all_letters = all_letters

        return all_letters, speaker_letters, listener_letters
        '''        if self.letter_opts['speaking_listening'] == 'both':
            return all_letters
        elif self.letter_opts['speaking_listening'] == 'listening':
            return listener_letters
        elif self.letter_opts['speaking_listening'] == 'speaking':
            return speaker_letters
        else:
            #print("Error: speaking_listening option is not valid. It should be 'both', 'listening' or 'speaking''.")
        '''

    def save_letters_to_file(self, all_letters, lettersFolder, letter_filename):
        if not os.path.exists(lettersFolder):
            os.makedirs(lettersFolder)
        outputfile_letters = (f"{lettersFolder}{letter_filename}")

        #print("writing letters to : ", outputfile_letters)
        letters = all_letters['letter']
        letters_header = ['frame_no', 'letter', 'magnitude', 'letter_yaw_component', 'letter_pitch_component',
                          'letter_roll_component', 'level', 'size', 'first_frame', 'last_frame', 'face_no', 'speaking',
                          'conf_part', 'fps']
        csv_file = open(outputfile_letters, 'w')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(letters_header)
        for i in range(len(letters)):
            csv_writer.writerow(
                [i, all_letters['letter'][i], all_letters['magnitude'][i], all_letters['letter_yaw_component'][i],
                 all_letters['letter_pitch_component'][i], all_letters['letter_roll_component'][i],
                 all_letters['level'][i], all_letters['size'][i], all_letters['first_frame'][i],
                 all_letters['last_frame'][i], all_letters['face_no'][i], all_letters['speaking'][i],
                 all_letters['conf_part'][i], all_letters['fps'][i]])
        csv_file.close()

    def get_letter_summary(self, all_letters, summaryFolder, summary_filename):
        # print if letters are not empty in one line
        if not os.path.exists(summaryFolder):
            os.makedirs(summaryFolder)

        letters = all_letters['letter']
        summary = ""
        #print("letters: ", end="")
        for i in range(len(letters)):
            if letters[i] != '' and letters[i] != ' ':
                summary = summary + str(letters[i])
                #print(letters[i], end="")
        #print("", end="\n")
        # save letters to a file
        filepath = f"{summaryFolder}{summary_filename}"
        summary_file = open(filepath, 'w')
        #print("writing letters to : ", filepath)
        summary_file.write(summary)
        summary_file.close()

    def get_letter_vectors(self):
        letter_vectors = []
        letter_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u','v', 'w', 'x', 'y', 'z','_']
        counter = 0
        for i in [1, -1, 0]:
            for j in [1, -1, 0]:
                for k in [1, -1, 0]:
                    class_id = letter_names[counter]
                    #print("i,j,k: ", i, j, k, "class_id: ", class_id)     #PYR
                    letter_vectors.append([i, j, k])
                    counter += 1
        #print("l: ", l)
        return letter_vectors, letter_names, dict(zip(letter_names, letter_vectors))
    def get_letter_name(self, pitch_response, yaw_response, roll_response, letter_vectors, letter_names):
        x = np.array([pitch_response, yaw_response, roll_response])
        x = x / np.linalg.norm(x)

        distances = []
        #print( "xT. x =", round(np.dot(x,x),1))
        for i in range(len(letter_vectors)):
            letter_vector = np.array(letter_vectors[i])
            if np.linalg.norm(letter_vector) > 0.0:
                letter_vector = letter_vector / np.linalg.norm(letter_vector)
            else:
                letter_vector = letter_vector / (np.linalg.norm(letter_vector) + 0.0001)
            distance = np.dot(x, letter_vector)
            distances.append(distance)
        max_index = distances.index(max(distances))
        #print ("max_index: ", max_index)
        letter = letter_names[max_index]
        #print("vector", x, "assigned vector: ", letter_vectors[max_index])
        #print("yaw_response, pitch_response, roll_response:  letter: ", x, letter, letter_vectors[max_index])
        return letter
    def no_motion_letters(self, no_motion_frames, letters, min_silence = 5, silent_block_size = 20 ):
        #find runs of no motion frames
        #Generate a "-" letter for each silent block
        silent_blocks=  []
        blocksize = 0
        zero_block_start_frame = []
        for i in range(len(no_motion_frames)):
            if no_motion_frames[i] == 0:
                blocksize = blocksize + 1
            else:
                silent_blocks.append(blocksize+1)
                zero_block_start_frame.append(max(i - blocksize,0))
                blocksize = 0

        for i in range(len(silent_blocks)):
            frame_no = zero_block_start_frame[i]
            if silent_blocks[i] > min_silence:
                #print("i , frame_no, silent_blocks[i]: ", i, frame_no, silent_blocks[i])
                for j in range(0, silent_blocks[i], silent_block_size):
                    letters[frame_no + j] = '-'
                    #print("letter '-' at silent_blocks[i]: ", frame_no + j)
        return letters
    def save_letters_to_db(self, all_letters, angles_file_name, databaseFile):
        # Connect to the SQLite database (it will be created if not exists)
        db_path = f"{databaseFile}"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        letter_table_name = 'letter_table'
        # Create the table if it does not exist
        create_table_query = f'''
        CREATE TABLE IF NOT EXISTS {letter_table_name} (
            file_name TEXT,
            frame_no INTEGER,
            letter TEXT,
            magnitude REAL,
            letter_yaw_component REAL,
            letter_pitch_component REAL,
            letter_roll_component REAL,
            level INTEGER,
            size REAL,
            first_frame INTEGER,
            last_frame INTEGER,
            face_no INTEGER,
            speaking INTEGER,
            conf_part INTEGER,
            fps REAL
        );
        '''
        cursor.execute(create_table_query)
        # Commit the changes to the database
        conn.commit()
        # Read the all_letter rows and insert data_in into the table
        for i in range(len(all_letters['frame_no'])):
            # Convert strings to appropriate data_in types based on the table schema
            row_data = {
                'filename': angles_file_name,
                'frame_no': int(all_letters['frame_no'][i]),
                'letter': all_letters['letter'][i],
                'magnitude': float(all_letters['magnitude'][i]),
                'letter_yaw_component': float(all_letters['letter_yaw_component'][i]),
                'letter_pitch_component': float(all_letters['letter_pitch_component'][i]),
                'letter_roll_component': float(all_letters['letter_roll_component'][i]),
                'level': all_letters['level'][i],
                'size': float(all_letters['size'][i]),
                'first_frame': int(all_letters['first_frame'][i]),
                'last_frame': int(all_letters['last_frame'][i]),
                'face_no': int(all_letters['face_no'][i]),
                'speaking': int(all_letters['speaking'][i]),
                'conf_part': int(all_letters['conf_part'][i]),
                'fps': float(all_letters['fps'][i])
                }
            # Insert data_in into the table
            insert_query = f'''
            INSERT INTO {letter_table_name} VALUES (
                :filename, :frame_no, :letter, :magnitude, :letter_yaw_component,
                :letter_pitch_component, :letter_roll_component, :level, :size,
                :first_frame, :last_frame, :face_no, :speaking, :conf_part, :fps
            );
            '''
            cursor.execute(insert_query, row_data)
        # Commit the changes to the database
        conn.commit()
        # Close the database connection
        conn.close()

def main():
    import cv2
    RootDir = '/Users/muhittingokmen/Dropbox/CHOP-MEF-data_in/monodic'
    videoFolder = f"{RootDir}/augmented_videos/"
    keypoint_folder = f"{RootDir}/keypoints/"
    #print("KeypointFolder: ", keypoint_folder)


    keypoints_yaw_folder = f"{keypoint_folder}yaw/"
    keypoints_pitch_folder = f"{keypoint_folder}pitch/"
    keypoints_roll_folder = f"{keypoint_folder}roll/"

    letterObj = Letters()

    letter_vectors, letter_names, letter_dict = letterObj.get_letter_vectors()

    videoFiles = sorted(os.listdir(videoFolder))
    #videoFiles = videoFiles[0:3]
    #print("videoFiles: ", videoFiles)
    # sys.exit()

    for videoFile in sorted(videoFiles):
        if not videoFile.endswith(".mp4"):
            continue
        video_file_path = f"{videoFolder}{videoFile}"
        videofile_base = os.path.splitext(videoFile)[0]
        # get ride off _augmented part
        #videofile_base = videofile_base.split("_")[0]
        videofile_base =  videofile_base.replace('_augmented', '')
        videofile_base = videofile_base.replace('_updated', '')

        filename = f"{videofile_base}_yaw_kines.csv"
        #print("input video filename: ", filename)
        keypoints_yaw_file = f"{keypoints_yaw_folder}{filename}"
        #print("KeypointFolder: ", keypoints_yaw_file)

        filename = f"{videofile_base}_pitch_kines.csv"
        keypoints_pitch_file = f"{keypoints_pitch_folder}{filename}"
        #print("KeypointFolder: ", keypoints_pitch_file)

        filename = f"{videofile_base}_roll_kines.csv"
        keypoints_roll_file = f"{keypoints_roll_folder}{filename}"
        #print("KeypointFolder: ", keypoints_roll_file)

        #speaker_filename = f"{SpeakerFolder}{videofile_base}_updated_labels.csv"

        keypoints_yaw = pd.read_csv(keypoints_yaw_file)
        keypoints_pitch = pd.read_csv(keypoints_pitch_file)
        keypoints_roll = pd.read_csv(keypoints_roll_file)
        #print("keypoints_yaw.speaking ", keypoints_yaw['speaking'][199])
        #get frame count of input video
        cap = cv2.VideoCapture(video_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = len (keypoints_yaw['position'])
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        min_silence = letterObj.min_silence
        silent_block_size = letterObj.silent_block_size
        #get letters
        # letters, letter_magnitudes, letter_yaw_component, letter_pitch_component, letter_roll_component,
        # letter_levels, letter_sizes, letter_first_frame, letter_last_frame, letter_face_no, letter_speaking,
        # letter_conf_part, letter_fps, video_frames_max = letterObj.get_letters(keypoints_yaw, keypoints_pitch,
        # keypoints_roll, frame_number, letter_vectors, letter_names, min_silence, silent_block_size)
        all_letters = letterObj.get_letters(
            keypoints_yaw, keypoints_pitch, keypoints_roll, frame_number, letter_vectors, letter_names, min_silence,
            silent_block_size)
        # save letters to a file
        letterObj.save_letters_to_file(all_letters, f"{RootDir}/letters/", f"{videofile_base}_letters.csv")
        letterObj.get_letter_summary(all_letters, f"{RootDir}/summary/", f"{videofile_base}_summary.txt")


if __name__ == "__main__":
    main()
