from parameters import *
from pygame import mixer
#Libraries used for Face Recognition
from scipy.spatial import distance
from imutils import face_utils as face
import imutils
import time
import dlib
import cv2
import queue
#Libraries used for suggesting nearby locations
import googlemaps
from geopy.geocoders import Nominatim
#Libraries used for Speech Recognition
import speech_recognition as sr
import pyaudio
import json
import spotipy
import webbrowser

#Face Outline
def face_area(captures):
    if len(captures)==0: 
        return
    captured_areas=[]
    for i in captures:
        captured_areas.append(i.area())
    max_index = captured_areas.index(max(captured_areas))
    return captures[max_index]

#Eye Aspect Ratio
def get_EAR(eye):
    ud_1 = distance.euclidean(eye[1], eye[5])
    ud_2 = distance.euclidean(eye[2], eye[4])
    lr = distance.euclidean(eye[0], eye[3])
    ear = (ud_1+ud_2)/(lr*2)
    return ear #aspect ratio of eye

#Mouth Aspect Ratio
def get_MAR(mouth):
    lr = distance.euclidean(mouth[0],mouth[4])
    ud = 0
    for mark in range(1,4):
        ud += distance.euclidean(mouth[mark],mouth[8-mark])
    mar = ud/(lr*3)
    return  mar#mouth aspect ratio

#Nearby rest areas
def nearby_locations():
    app = Nominatim(user_agent="test")
    address = "UTA Blvd, Arlington, TX"
    location = app.geocode(address).raw
    latitude = location['lat']
    longitude = location['lon']

    client = googlemaps.Client(key = "AIzaSyD_D3J-yYUiSknK7DP0mpYjdGR9TuLih9o")
    fields = ['place_id', 'name', 'formatted_address', 'business_status']
    location_bias = 'circle:{}@{},{}'.format(5000, latitude, longitude) 
    place_details = client.find_place(input = 'cafe',
                                      input_type = 'textquery',
                                      location_bias = location_bias, 
                                      fields = fields)
    places = client.places_nearby(location=(latitude, longitude), radius = 5000, type='cafe')
    for i in places['results']:
      print(i['name'],i['vicinity'])
#Mic
def mic():
  #print("In mic")
  r = sr.Recognizer()
  my_mic = sr.Microphone()
  with my_mic as source:
    audio_text = r.listen(source, 5)
    try:
      text = r.recognize_google(audio_text)   
      return text  
    except:
      text = "invalid"
      return text

#Open Spotify
def openSpotify():
  username = '315cokow7oxy4mypmdecuomlhsdq'
  clientID = 'de24801f24294e59926801387a914903'
  clientSecret = 'f97bcc1dbdb04df48c2fcd5598e0eff1'
  redirect_uri = 'http://google.com/'
  oauth_object = spotipy.SpotifyOAuth(clientID, clientSecret, redirect_uri)
  token_dict = oauth_object.get_cached_token()
  token = token_dict['access_token']
  spotifyObject = spotipy.Spotify(auth=token)
  user_name = spotifyObject.current_user()
  while True:
    search_song = 'Cradles'
    results = spotifyObject.search(search_song, 1, 0, "track")
    songs_dict = results['tracks']
    song_items = songs_dict['items']
    song = song_items[0]['external_urls']['spotify']
    webbrowser.open(song)
    print('Opened Spotify')
    break
    
#Play music
def music():
  mixer.music.load(alarm_paths[3])
  mixer.music.play()
  print("Would you like me to play some music?")
  time.sleep(1.8)
  driver_ip2 = mic()
  print(driver_ip2)
  if(driver_ip2 == 'yes'):
    print("Playing music")
    openSpotify()
    print("Played music")
  elif(driver_ip2 == 'no'):
    print("Okay")

#Alert function
def break_call(alarm_type):
    mixer.music.load(alarm_paths[alarm_type])
    mixer.music.play()
    time.sleep(2.5)
    print("Do you want a coffee break?")
    driver_ip = mic()
    if(driver_ip == 'yes'):
        nearby_locations()
    elif(driver_ip == 'no'):
        music()	
    else:
      return None

# Facial processing
def facial_processing():
    mixer.init()
    eye_status = False
    mouth_status = False
    distraction_status = False
    yawn_count = 0

    face_detector = dlib.get_frontal_face_detector()
    shape_prediction = dlib.shape_predictor(shape_predictor_path)

    ls,le = face.FACIAL_LANDMARKS_IDXS["left_eye"]
    rs,re = face.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap=cv2.VideoCapture(0)

    fps_couter=0
    fps_to_display='initializing...'
    fps_timer=time.time()
    while True:
        _ , frame=cap.read()
        fps_couter+=1
        frame = cv2.flip(frame, 1)
        if time.time()-fps_timer>=1.0:
            fps_to_display=fps_couter
            fps_timer=time.time()
            fps_couter=0
        cv2.putText(frame, "FPS :"+str(fps_to_display), (frame.shape[1]-100, frame.shape[0]-10),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        #frame = imutils.resize(frame, width=900)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_landmarks = face_detector(gray, 0)
        face_user = face_area(face_landmarks)

        if face_user!=None:

            distraction_status=False

            shape_new = shape_prediction(gray, face_user)
            shape_new = face.shape_to_np(shape_new)

            left_eye = shape_new[ls:le]
            right_eye = shape_new[rs:re]
            EAR_left = get_EAR(left_eye)
            EAR_right = get_EAR(right_eye)
            total_EAR = (EAR_left + EAR_right) / 2.0
            inner_lips=shape_new[60:68]
            MAR=get_MAR(inner_lips)


            left_eyeHull = cv2.convexHull(left_eye)
            right_eyeHull = cv2.convexHull(right_eye)
            mouth_Hull = cv2.convexHull(inner_lips)
            cv2.drawContours(frame, [left_eyeHull], -1, (255, 255, 255), 1)
            cv2.drawContours(frame, [right_eyeHull], -1, (255, 255, 255), 1)
            cv2.drawContours(frame, [mouth_Hull], -1, (255, 255, 255), 1)
            cv2.putText(frame, "EAR: {:.2f} MAR{:.2f}".format(total_EAR,MAR), (10, frame.shape[0]-10),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if total_EAR < EYE_DROWSINESS_THRESHOLD:

                if not eye_status:
                    eye_start_time= time.time()
                    eye_status=True

                if time.time()-eye_start_time >= EYE_DROWSINESS_INTERVAL:
                    cv2.putText(frame, "YOU SEEM SLEEPY...PLEASE TAKE A BREAK!", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if  not distraction_status and not mouth_status and not mixer.music.get_busy():
                        mixer.music.load(alarm_paths[0])
                        mixer.music.play()
                    
            else:
                eye_status=False
                if not distraction_status and not mouth_status and mixer.music.get_busy():
                    mixer.music.stop()


            if MAR > MOUTH_DROWSINESS_THRESHOLD:

                if not mouth_status:
                    mouth_start_time= time.time()
                    mouth_status=True
                    

                if time.time()-mouth_start_time >= MOUTH_DROWSINESS_INTERVAL:
                    alarm_type=2
                    cv2.putText(frame, "YOU ARE YAWNING", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    yawn_count = yawn_count + 1
                    mouth_status=False
                    print("Yawn Count: ",yawn_count)
                
       
                if yawn_count > 2 and not mixer.music.get_busy():
                    break_call(2)
                    yawn_count = 0
                         
            else:
                mouth_status=False
                if not distraction_status and not eye_status and mixer.music.get_busy():
                    mixer.music.stop()       


        else:
            
            if not distraction_status:
                distracton_start_time=time.time()
                distraction_status=True

            if time.time()- distracton_start_time> DISTRACTION_INTERVAL:
                

                cv2.putText(frame, "EYES ON ROAD", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if yawn_count <= 2 and not eye_status and not mouth_status and not  mixer.music.get_busy():
                    mixer.music.load(alarm_paths[1])
                    mixer.music.play()
                         

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(5)&0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__=='__main__':
	facial_processing()


