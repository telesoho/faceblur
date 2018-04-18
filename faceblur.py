'''
Recognize and blur all faces in photos.
'''
import os
import sys
import cv2
import face_recognition

def face_blur(src_img, dest_img, zoom_in=1):
    '''
    Recognize and blur all faces in the source image file, then save as destination image file.
    '''
    sys.stdout.write("%s:processing... \r" % (src_img))
    sys.stdout.flush()

    # Initialize some variables
    face_locations = []
    photo = face_recognition.load_image_file(src_img)
    # Resize image to  1/zoom_in size for faster face detection processing
    small_photo = cv2.resize(photo, (0, 0), fx=1/zoom_in, fy=1/zoom_in)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_photo, model="cnn")

    if face_locations:
        print("%s:There are %s faces at " % (src_img, len(face_locations)), face_locations)
    else:
        print('%s:There are no any face.' % (src_img))
        return False

    #Blur all face
    photo = cv2.imread(src_img)
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/zoom_in size
        top *= zoom_in
        right *= zoom_in
        bottom *= zoom_in
        left *= zoom_in

        # Extract the region of the image that contains the face
        face_image = photo[top:bottom, left:right]

        # Blur the face image
        face_image = cv2.GaussianBlur(face_image, (21, 21), 0)

        # Put the blurred face region back into the frame image
        photo[top:bottom, left:right] = face_image

    #Save image to file
    cv2.imwrite(dest_img, photo)

    print('Face blurred photo has been save in %s' % dest_img)

    return True

def blur_all_photo(src_dir, dest_dir):
    '''
    Blur all faces in the source directory photos and copy them to destination directory
    '''
    src_dir = os.path.abspath(src_dir)
    dest_dir = os.path.abspath(dest_dir)
    print('Search and blur human faces in %s''s photo.' % src_dir)
    for root, subdirs, files in os.walk(src_dir):
        root_relpath = os.path.relpath(root, src_dir)
        new_root_path = os.path.realpath(os.path.join(dest_dir, root_relpath))
        os.makedirs(new_root_path, exist_ok=True)

        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext == '.jpg':
                srcfile_path = os.path.join(root, filename)
                destfile_path = os.path.join(new_root_path, filename)
                face_blur(srcfile_path, destfile_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('faceblur v1.0.0 (c) telesoho.com')
        print('Usage:python faceblur <src-image/src-directory> <dest-image/dest-directory>')
    else:
        if os.path.isfile(sys.argv[1]):
            face_blur(sys.argv[1], sys.argv[2])
        else:
            blur_all_photo(sys.argv[1], sys.argv[2])
