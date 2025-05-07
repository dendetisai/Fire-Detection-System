import smtplib
from email.mime.text import MIMEText
import threading
import cv2
import serial  
from twilio.rest import Client
from ultralytics import YOLO
import torch
import time

# Configure Arduino Serial Communication
arduino = serial.Serial('COM10', 9600, timeout=1)

# Email setup
sender = 'sender_email'
password = 'senders_email_passcode'
receivers = ['recivers list']
subject = '🔥 Fire Detected Alert!'
message = '🚨 Fire accident detected! Take immediate action.'

# Load YOLO model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(r"C:\Users\jeevan sai\Downloads\best.pt").to(device)

# Twilio API credentials
account_sid = "********************************"
auth_token = "*********************************"
client = Client(account_sid, auth_token)

to_phone = "+91**********"
from_phone = "+12********"
twilio_voice_url = "http://demo.twilio.com/docs/voice.xml"

# Fire detection flags
call_email_alert = False
last_fire_time = None
timeout_duration = 120

# Buzzer control
buzzer_on = False
buzzer_duration = 20

def send_email():
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(receivers)
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receivers, msg.as_string())
        print("📧 Fire alert email sent!")
    except Exception:
        pass

def make_call():
    try:
        client.calls.create(url=twilio_voice_url, to=to_phone, from_=from_phone)
        print(f"📞 Emergency call placed to {to_phone}!")
    except Exception as e:
        print(f"[ERROR] Call failed: {e}")

def send_signal_to_arduino(state):
    signal = b"HIGH\n" if state else b"LOW\n"
    arduino.write(signal)
    arduino.flush()

def turn_off_buzzer():
    global buzzer_on
    send_signal_to_arduino(False)
    buzzer_on = False

def fire_detection(frame):
    results = model(frame, verbose=False)
    fire_detected = False
    
    for result in results:
        for box in result.boxes:
            b = box.xyxy[0].cpu().numpy()
            c = int(box.cls)
            conf = float(box.conf.cpu().numpy())
            
            if c == 0 and conf > 0.70:
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
                cv2.putText(frame, f"🔥 Fire {conf:.2f}", 
                            (int(b[0]), int(b[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                fire_detected = True
    
    return fire_detected, frame

def Alert_Processor(fire_detected):
    global call_email_alert, last_fire_time, buzzer_on
    current_time = time.time()
    
    if fire_detected:
        print("🔥 Fire detected!")
        last_fire_time = current_time
        
        if not call_email_alert:
            call_email_alert = True
            threading.Thread(target=send_email, daemon=True).start()
            threading.Thread(target=make_call, daemon=True).start()
        
        if not buzzer_on:
            buzzer_on = True
            threading.Thread(target=send_signal_to_arduino, args=(True,), daemon=True).start()
            threading.Timer(buzzer_duration, turn_off_buzzer).start()
    
    if last_fire_time and (current_time - last_fire_time >= timeout_duration):
        print("✅ No fire detected for 2 minutes. System resets alert flags for the next fire event..")
        call_email_alert = False
        last_fire_time = None
        send_signal_to_arduino(False)

def main():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (640, 480))
        fire_detected, frame_processed = fire_detection(frame_resized)
        
        Alert_Processor(fire_detected)
        
        cv2.imshow("🔥 YOLOv8 Fire Detection", frame_processed)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()