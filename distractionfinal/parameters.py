import os

shape_predictor_path    = os.path.join('data','shape_predictor_68_face_landmarks.dat')
alarm_paths             = [os.path.join('data','audio_files','sleepy.wav'),
                           os.path.join('data','audio_files','eyes_on_road.wav'),
                           os.path.join('data','audio_files','sleepy_break.wav'),
                           os.path.join('data','audio_files','music_ask.wav')]

EYE_DROWSINESS_THRESHOLD    = 0.25
EYE_DROWSINESS_INTERVAL     = 2.0
MOUTH_DROWSINESS_THRESHOLD  = 0.37
MOUTH_DROWSINESS_INTERVAL   = 1.5
DISTRACTION_INTERVAL        = 2.0
