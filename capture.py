import cv2, time, datetime
from  pandas import DataFrame as df

status_lis = [None, None]
times = []
firstFrame = None
timefill = df(columns=['start', 'end'])
video = cv2.VideoCapture(0)
while True:
    check, frame = video.read()
    status = 0
    gray_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_vid = cv2.GaussianBlur(gray_vid, (21, 21), 0)
    if firstFrame is None:
        firstFrame = gray_vid

    delta_frame = cv2.absdiff(firstFrame, gray_vid)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)  # for clear image this is optional
    (_, cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 5)
    status_lis.append(status)
    status_lis = status_lis[-2:]

    if status_lis[-1] == 0 and status_lis[-2] == 1:
        times.append(datetime.datetime.now())
    if status_lis[-1] == 1 and status_lis[-2] == 0:
        times.append(datetime.datetime.now())
    cv2.imshow("video", frame)
    # cv2.imshow("delta", delta_frame)
    # cv2.imshow("thresh", thresh_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            times.append(datetime.datetime.now())
        break
for i in range(0, len(times), 2):
    timefill = timefill.append({'start': times[i], 'end': times[i + 1]}, ignore_index=True)

timefill.to_csv("times.csv")
video.release()
cv2.destroyAllWindows()
