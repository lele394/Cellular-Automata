

import cv2
import os









def MakeVideo(image_folder, video_name, fps):
    #image_folder = 'temp'
    #video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".bmp")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()





def SaveImage(frame, folder, name):
    frame2 = cv2.normalize(frame, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    cv2.imwrite(folder+str(name)+".jpg", frame2)
