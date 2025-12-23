import os

import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26  # A-Z letters
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    letter = chr(65 + j)  # Convert to A-Z
    print('Collecting data for class {} - Letter {}'.format(j, letter))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready for letter {}? Press "Q"'.format(letter), (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Class {}/26'.format(j+1), (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.putText(frame, 'Collecting: {}/{}'.format(counter, dataset_size), (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Letter: {}'.format(letter), (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1
    
    print('Completed class {} - Letter {}! ({} images)'.format(j, letter, dataset_size))

cap.release()
cv2.destroyAllWindows()
