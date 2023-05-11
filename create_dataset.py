import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# gesture = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f',
#            6:'g', 7:'h', 8:'i', 9:'k', 10:'l', 11:'m',
#            12:'n', 13:'o', 14:'p', 15:'q', 16:'r', 17:'s',
#            18:'t', 19:'u', 20:'v', 21:'w', 22:'x', 23:'y'}
idx = 19

cap = cv2.VideoCapture(0)
# 비디오의 너비와 높이 설정
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

                # 30프레임이 지날 때마다 랜드마크 좌표를 출력합니다.
                if cv2.waitKey(1) == ord('a'):
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) % 30 == 0:
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
                                angle = np.append(angle, idx)
                                data = np.array([angle], dtype=np.float32)
                                
                                #npy 형태로 저장하는 코드구현 
                                #np.save('./sltranslation/dataset/a_data.npy', data)

                                #csv 형태로 저장하는 코드구현
                                df = pd.DataFrame(data)
                                df.to_csv('./sltranslation/dataset/u_data.csv', mode='a', header=False, index=False)
        #좌우 반전
        cv2.imshow('Sign language translation', cv2.flip(image, 1))

        key = cv2.waitKey(1) & 0xFF
        if (key == 27):
            break
cap.release()