import cv2
import os

def has_img_face(img):
    # Load the pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = img

    # Convert the image to grayscale (face detection works better in grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        return True
    return False


def detect_faces(img, count, timestamp, videoname, folder_path):
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    timestamp = timestamp * count
    formatted_time = "{:.2f}".format(timestamp)

    print("[INFO] Found {0} Faces.".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")

        target_size = (48, 48)  # Adjust to the desired size
        resized_img = cv2.resize(roi_color, target_size)
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        print("Image resizing complete.")
        # kaydetme işlemi oluşturulan dosya dizinine yapılacaktır # to do
        videoname = videoname.rsplit('/')[-1]
        cv2.imwrite(str(folder_path)+'/'+str(videoname) + '_' + str(formatted_time) + '_faces.jpg', gray_img)


def create_folder(folder_name):
    # Specify the path of the folder
    folder_path = r'/videos/'+folder_name

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    return folder_path


def extract(video_file_path):
    # path of video
    #videopath = r"/Users/a2023/Desktop/GraduationProject/Code/videos/buzdolabi.mp4"
    videopath = video_file_path
    videoname = videopath.rsplit('//', 1)[-1]
    #.mp4 kısmı için gerekirse split işlemi yapabiliriz.
    split = videoname.rsplit('/')
    split = split[-1]
    folder_path = create_folder(split)
    vidcap = cv2.VideoCapture(videopath)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    time_stamp = duration / frame_count

    success, image = vidcap.read()

    count = 0
    try:
        while vidcap.isOpened():
            success, image = vidcap.read()
            if has_img_face(image):
                # cv2.imwrite(r"C:\Users\User\Desktop\Graduation Project\Code\videos\frames\%d.png" % count, image)
                detect_faces(image, count, time_stamp, videoname, folder_path)
            count += 1
        return folder_path
    except Exception as e:
        return folder_path
