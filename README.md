import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
#  اطلاعات مربوط به قد و وزن و جنسیت
g=input('enter your gender if male type m if female type f: ')
m=float(input('enter your weight in kg : '))
h=float(input('enter your height in cm : '))
fr=float(input('enter the dumbbell in your right hand in N : '))
fl=float(input('enter the dumbbell in your left hand in N : '))

# طول ها و وزن های مورد نیاز از جدول در کتاب سوزان
# Segment Lengths
datasl = {
    'SEGMENT': ['Head and neck', 'Trunk', 'Upper arm', 'Forearm', 'Hand', 'Thigh', 'Lower leg', 'Foot'],
    'MALES': [10.75, 30.00, 17.20, 15.70, 5.75, 23.20, 24.70, 4.25],
    'FEMALES': [10.75, 29.00, 17.30, 16.00, 5.75, 24.90, 25.70, 4.25]
}

sl = pd.DataFrame(datasl)

# Segment Weights


datawi = {
    'SEGMENT': ['Head', 'Trunk', 'Upper arm', 'Forearm', 'Hand', 'Thigh', 'Lower leg', 'Foot'],
    'MALES': [8.26, 46.84, 3.25, 1.87, 0.65, 10.50, 4.75, 1.43],
    'FEMALES': [8.20, 45.00, 2.90, 1.57, 0.50, 11.75, 5.35, 1.33]
}

wi = pd.DataFrame(datawi)

# Segmental Center of Gravity Locations

datacg = {
    'SEGMENT': ['Head and neck', 'Trunk', 'Upper arm', 'Forearm', 'Hand', 'Thigh', 'Lower leg', 'Foot'],
    'MALES': [55.0, 63.0, 43.6, 43.0, 46.8, 43.3, 43.4, 50.0],
    'FEMALES': [55.0, 56.9, 45.8, 43.4, 46.8, 42.8, 41.9, 50.0]
}

cg = pd.DataFrame(datacg)

#Segmental Radii of Gyration Measured from Proximal and Distal Segment Ends

dataIC = {
    'SEGMENT': ['Upper arm', 'Forearm', 'Hand', 'Thigh', 'Lower leg', 'Foot'],
    'MALES_PROXIMAL': [54.2, 52.6, 54.9, 54.0, 52.9, 69.0],
    'MALES_DISTAL': [64.5, 54.7, 54.9, 65.3, 64.2, 69.0],
    'FEMALES_PROXIMAL': [56.4, 53.0, 54.9, 53.5, 51.4, 69.0],
    'FEMALES_DISTAL': [62.3, 64.3, 54.9, 65.8, 65.7, 69.0]
}

IC= pd.DataFrame(dataIC)
# داده های مورد نیاز برای محاسبات نیرو و گشتاور در دست :
if g=='f' :
    slaup=h*0.001*sl.loc[sl['SEGMENT'] == 'Upper arm', 'FEMALES'].values[0]/100
    slafm=h*0.001*sl.loc[sl['SEGMENT'] == 'Forearm', 'FEMALES'].values[0]/100
    wiaup=m*wi.loc[wi['SEGMENT'] == 'Upper arm', 'FEMALES'].values[0]/100
    wiafm=m*wi.loc[wi['SEGMENT'] == 'Forearm', 'FEMALES'].values[0]/100
    cgaup=slaup*cg.loc[cg['SEGMENT'] == 'Upper arm', 'FEMALES'].values[0]/100
    cgafm=slafm*cg.loc[cg['SEGMENT'] == 'Forearm', 'FEMALES'].values[0]/100
    icaup=slaup*IC.loc[IC['SEGMENT'] == 'Upper arm', 'FEMALES_PROXIMAL'].values[0]/100
    icafm=slafm*IC.loc[IC['SEGMENT'] == 'Forearm', 'FEMALES_PROXIMAL'].values[0]/100
elif g=='m' :
    slaup=h*0.001*sl.loc[sl['SEGMENT'] == 'Upper arm', 'MALES'].values[0]/100
    slafm=h*0.001*sl.loc[sl['SEGMENT'] == 'Forearm', 'MALES'].values[0]/100
    wiaup=m*wi.loc[wi['SEGMENT'] == 'Upper arm', 'MALES'].values[0]/100
    wiafm=m*wi.loc[wi['SEGMENT'] == 'Forearm', 'MALES'].values[0]/100
    cgaup=slaup*cg.loc[cg['SEGMENT'] == 'Upper arm', 'MALES'].values[0]/100
    cgafm=slafm*cg.loc[cg['SEGMENT'] == 'Forearm', 'MALES'].values[0]/100 
    icaup=slaup*IC.loc[IC['SEGMENT'] == 'Upper arm', 'MALES_PROXIMAL'].values[0]/100 -wiaup*cgaup**2
    icafm=slafm*IC.loc[IC['SEGMENT'] == 'Forearm', 'MALES_PROXIMAL'].values[0]/100 -wiafm*cgafm**2



# mediapipe start
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
anglesr = []
anglesl = []

# تابع محاسبه زاویه با مبنای زاویه صفر پایین
def calculate_angle_with_base(a, b, c, base_point):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    base = np.array(base_point)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.degrees(np.abs(radians))

    # اعمال مبنای زاویه صفر پایین
    if b[1] < base[1]:
        angle = 120 - angle 
    
    return angle

# ثبت زمان آغازین
start_time = time.time()
current_time = start_time

cap = cv2.VideoCapture(0)

# ابتدایی‌کردن مدل مدیاپایپ
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    prev_angle_right = 0
    prev_angle_left = 0
    prev_angular_speed_right = 0
    prev_angular_speed_left = 0
    acceleration_right = 0
    acceleration_left = 0
    angular_speed_right_list = []
    angular_speed_left_list = []
    acceleration_right_list = []
    acceleration_left_list = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # تبدیل تصویر به RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # تشخیص اجزای بدن
        results = pose.process(image)

        # تبدیل به تصویر BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # استخراج لندمارک‌ها
        try:
            landmarks = results.pose_landmarks.landmark

            # دست راست
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # دست چپ
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # محاسبه زاویه برای دست راست و دست چپ با مبنای زاویه صفر پایین
            angle_right = calculate_angle_with_base(right_shoulder, right_elbow, right_wrist, right_wrist)
            angle_left = calculate_angle_with_base(left_shoulder, left_elbow, left_wrist, left_wrist)

            # محاسبه سرعت زاویه‌ای به رادیان بر ثانیه
            current_time = time.time()
            time_difference = current_time - last_time
            angular_speed_right = (angle_right - prev_angle_right)*3.14/180 / time_difference if time_difference > 0 else 0
            angular_speed_left = (angle_left - prev_angle_left)*3.14/180 / time_difference if time_difference > 0 else 0
            # save angles
            anglesr.append(angle_right)
            anglesl.append(angle_left)

            # اعمال فیلتر برای کاهش نوسانات
            angular_speed_right = 0.9 * prev_angular_speed_right + 0.1 * angular_speed_right
            angular_speed_left = 0.9 * prev_angular_speed_left + 0.1 * angular_speed_left

            # محاسبه شتاب برحسب رادیان بر مجذور ثانیه
            acceleration_right = (angular_speed_right - prev_angular_speed_right) / time_difference if time_difference > 0 else 0
            acceleration_left = (angular_speed_left - prev_angular_speed_left) / time_difference if time_difference > 0 else 0

            # نشان دادن زاویه روی دست
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, f"{angle_right:.2f} deg",
                        tuple(np.multiply(right_wrist, [640, 480]).astype(int)),
                        font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"{angle_left:.2f} deg",
                        tuple(np.multiply(left_wrist, [640, 480]).astype(int)),
                        font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Angular Speed _right: {angular_speed_right:.2f} rad/s",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f" Angular Speed _left: {angular_speed_left:.2f} rad/s",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f" Acceleration _right: {acceleration_right:.2f} rad/s^2",
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f" Acceleration _left: {acceleration_left:.2f} rad/s^2",
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # نشان دادن خطوط پوزیشن
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            # ذخیره مقادیر شتاب و سرعت هر 0.5 ثانیه
            if current_time - start_time >= 0.5:
                angular_speed_right_list.append(angular_speed_right)
                angular_speed_left_list.append(angular_speed_left)
                acceleration_right_list.append(acceleration_right)
                acceleration_left_list.append(acceleration_left)
                start_time = current_time

            # به‌روزرسانی متغیرهای پیشین
            prev_angle_right = angle_right
            prev_angle_left = angle_left
            prev_angular_speed_right = angular_speed_right
            prev_angular_speed_left = angular_speed_left

        except Exception as e:
            print(e)

        # نمایش تصویر
        cv2.imshow('Webcam Feed', image)

        # به‌روزرسانی زمان آخرین
        last_time = current_time

        # بستن خودکار پنجره پس از 10 ثانیه
        if current_time - start_time > 10:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# چاپ مقادیر شتاب و سرعت هر 0.5 ثانیه
print("Angular Speed Right (rad/s):", angular_speed_right_list)
print("Angular Speed Left (rad/s):", angular_speed_left_list)
print("Acceleration Right (rad/s^2):", acceleration_right_list)
print("Acceleration Left (rad/s^2):", acceleration_left_list)


# forearm force and torque Right hand
FxRf=wiafm*(np.mean(acceleration_right_list)*slafm*np.cos(np.mean(anglesr))-(np.mean(angular_speed_right_list))**2*slafm*np.sin(np.mean(anglesr)))
FyRf=wiafm*(-np.mean(acceleration_right_list)*slafm*np.sin(np.mean(anglesr))-(np.mean(angular_speed_right_list))**2*slafm*np.cos(np.mean(anglesr)))+fr+wiafm*9.81
tfR=fr*slafm+cgafm*wiafm*9.81+icafm*np.mean(acceleration_right_list)

# forearm force and torque left hand
Fxlf=wiafm*(np.mean(acceleration_left_list)*slafm*np.cos(np.mean(anglesr))-(np.mean(angular_speed_left_list))**2*slafm*np.sin(np.mean(anglesr)))
Fylf=wiafm*(-np.mean(acceleration_left_list)*slafm*np.sin(np.mean(anglesr))-(np.mean(angular_speed_left_list))**2*slafm*np.cos(np.mean(anglesr)))+fl+wiafm*9.81
tfl=fr*slafm+cgafm*wiafm*9.81+icafm*np.mean(acceleration_left_list)

# uperarm right
FxRu=FxRf
FyRu=FyRf+wiaup*9.81
tuR=tfR

#upperarm left
FxLu=Fxlf
FyLu=Fylf+wiaup
tuL=tfl


print("Fx in righthand forearm(N):",FxRf )
print("Fy in righthand forearm(N):", FyRf)
print("torque in righthand forearm(N.m):", tfR)
print("Fx in lefthand forearm(N):",Fxlf )
print("Fy in left forearm(N):", Fylf)
print("torque in left forearm(N.m):", tfl)

print("Fx in righthand upperarm(N):",FxRf )
print("Fy in righthand upperarm(N):", FyRf)
print("torque in righthand upperarm(N.m)", tfR)
print("Fx in lefthand upperarm(N):",Fxlf )
print("Fy in left upperarm(N):", Fylf)
print("torque in left upperarm(N.m):", tfl)



##########  بخش رسم و نیروی لحظهای برای دست راست
FFxRf = []
for i in range(1, len(acceleration_right_list) + 1):
    FFxRf.append(wiafm * (acceleration_right_list[i - 1] * slafm * np.cos(anglesr[i - 1]) - 
                          (angular_speed_right_list[i - 1])**2 * slafm * np.sin(anglesr[i - 1])))

FFyRf = []
for i in range(1, len(acceleration_right_list) + 1):
    FFyRf.append(wiafm * (-acceleration_right_list[i - 1] * slafm * np.sin(anglesr[i - 1]) - 
                          (angular_speed_right_list[i - 1])**2 * slafm * np.cos(anglesr[i - 1]))+fr+wiafm*9.81)

ttRf = []
for i in range(1, len(acceleration_right_list) + 1):
    ttRf.append(fr*slafm+cgafm*wiafm*9.81+icafm*acceleration_right_list[i-1])

        
time_values1 = range(1, len(angular_speed_right_list) + 1)

time_values2 = range(1, len(acceleration_right_list) + 1)
# ایجاد یک صفحه و زیرنمودارها
fig, axs = plt.subplots(5, 1, figsize=(10, 12))

# نمودار اول
axs[0].plot(time_values1, angular_speed_right_list, marker='o')
axs[0].set_xlabel('time(s)')
axs[0].set_ylabel('velocity(rad/s)')
axs[0].set_title('Velocity vs. Time')

# نمودار دوم
axs[1].plot(time_values1, acceleration_right_list, marker='o')
axs[1].set_xlabel('time(s)')
axs[1].set_ylabel('acceleration(rad/s^2)')
axs[1].set_title('Acceleration vs. Time')

# نمودار سوم
axs[2].plot(time_values2, FFxRf, marker='o')
axs[2].set_xlabel('time(s)')
axs[2].set_ylabel('FX(N)')
axs[2].set_title('FX vs. Time')

# نمودار چهارم
axs[3].plot(time_values2, FFyRf, marker='o')
axs[3].set_xlabel('time(s)')
axs[3].set_ylabel('Fy(N)')
axs[3].set_title('Fy vs. Time')

# نمودار پنجم
axs[4].plot(time_values2, ttRf, marker='o')
axs[4].set_xlabel('time(s)')
axs[4].set_ylabel('torque(N.m)')
axs[4].set_title('FX vs. Time')

# تنظیمات نهایی
plt.tight_layout()  # تضمین چیدمان مناسب زیرنمودارها
plt.show()
