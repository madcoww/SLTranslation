import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions= ['e', 'i', 'l', 'o', 'u', 'v', 'y']

action_seq = []
this_action = '?'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model = load_model("./sltranslation/model/59model.hdf5")

cap = cv2.VideoCapture(0)
# 비디오의 너비와 높이 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
        # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
            continue

        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # 이미지에 손 주석을 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                if results.multi_hand_landmarks:
                                for hand_landmarks in results.multi_hand_landmarks:
                                    joint = np.zeros((21, 3))
                                    for j, lm in enumerate(hand_landmarks.landmark):
                                        joint[j] = [lm.x, lm.y, lm.z] # x, y, z 좌표를 저장합니다.

                                # 각도를 구합니다.    
                                    v1 = joint[[0, 1 ,2 ,3 ,0 ,5 ,6 ,7 ,0 ,9 ,10 ,11 ,0 ,13 ,14 ,15 ,0 ,17 ,18 ,19], :]
                                    v2 = joint[[1, 2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14 ,15 ,16 ,17 ,18 ,19 ,20], :]
                                    v = v2 - v1
                                # 벡터의 크기를 1로 만듭니다.
                                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                                
                                    angle = np.arccos(np.einsum('nt,nt->n',
                                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10 ,12 ,13 ,14 ,16 ,17 ,18], :],
                                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                                    angle = np.degrees(angle)

                                    #label 작업 제외 15개의 데이터

                                    input_data = np.expand_dims(np.array(angle, dtype=np.float32), axis=0)

                                    y_pred = model.predict(input_data).squeeze()

                                    i_pred = int(np.argmax(y_pred)) #argmax : softmax 변환
                                    conf = y_pred[i_pred]

                                    if conf < 0.95 : 
                                        continue
                                
                                    action = actions[i_pred]

                                    action_seq.append(action)

                                    if len(action_seq) < 3:
                                        continue
                                     
                                    if action_seq[-1] == action_seq[-2] == action_seq[-3] :
                                        this_action = action
                                    


                                    cv2.putText(image, this_action, org=(int(frame_width/2), int(frame_height*4/5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                                
        #좌우 반전
        
        cv2.imshow('Sign language translation', image)
        #cv2.imshow('Sign language translation', cv2.flip(image, 1))

        key = cv2.waitKey(1) & 0xFF
        if (key == 27):
            break
cap.release()