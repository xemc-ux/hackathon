import cv2
import time

cap = cv2.VideoCapture(0)
timeBadLighting = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = cv2.mean(gray)[0]
    
    is_dim = avg_brightness < 50
    
    if not is_dim:
        light_text = "LIGHTING: OK"
    else:
        light_text = "TOO DARK: TURN ON A LIGHT"
        checkElapsed = time.time() - timeBadLighting
        if checkElapsed > 10:
            status = "BAD LIGHTING! please correct!"
            print("send notif. you are in a dark room.")
    l_color = (0, 255, 0) if not is_dim else (0, 165, 255)
    
    cv2.putText(frame, light_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, l_color, 2)
    cv2.putText(frame, f"Brightness: {int(avg_brightness)}", (20, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Lighting Monitor', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
