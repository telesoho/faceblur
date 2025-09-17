'''
Recognize and blur all faces in photos.
'''
import argparse
import cv2
import face_recognition
import os
import sys
from shutil import copyfile

def face_blur(src_img, dest_img, args):
    '''
    Recognize and blur all faces in the source image file, then save as destination image file.
    '''
    sys.stdout.write("%s:processing... \r" % (src_img))
    sys.stdout.flush()

    # Initialize some variables
    face_locations = []
    photo = face_recognition.load_image_file(src_img)
    # Resize image to  1/zoom_in size for faster face detection processing
    small_photo = cv2.resize(photo, (0, 0), fx=1/args.zoom_in, fy=1/args.zoom_in)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_photo, model=args.model, number_of_times_to_upsample=args.upsampling)

    if face_locations:
        print("%s:There are %s faces at " % (src_img, len(face_locations)), face_locations)
    else:
        print('%s:There are no any face.' % (src_img))
        if args.copy:
            copyfile(src_img, dest_img)
            print('Original photo has been saved in %s' % dest_img)
        return False

    #Blur all face
    photo = cv2.imread(src_img)
    for top, right, bottom, left in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/zoom_in size
        top *= args.zoom_in
        right *= args.zoom_in
        bottom *= args.zoom_in
        left *= args.zoom_in

        # Extract the region of the image that contains the face
        face_image = photo[top:bottom, left:right]

        # Blur the face image
        face_image = cv2.GaussianBlur(face_image, (args.blurr, args.blurr), 0)

        # Put the blurred face region back into the frame image
        photo[top:bottom, left:right] = face_image

    #Save image to file
    cv2.imwrite(dest_img, photo)

    print('Face blurred photo has been saved in %s' % dest_img)

    return True

def blur_all_photo(src_dir, dest_dir, args):
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
                destfile_path = os.path.join(new_root_path, os.path.basename(filename))
                face_blur(srcfile_path, destfile_path, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Recognize and blur all faces in photo. faceblur v1.0.0 (c) telesoho.com''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src', help='source image or directory')
    parser.add_argument('dest', help='destination image or directory')
    parser.add_argument('-c', '--copy', action="store_true", default=False, help='copy src image to dest even if no faces have been detected')
    parser.add_argument('-m', '--model', default='cnn', choices=['cnn', 'hog'], help='recognition model, Convolutional Neural Network (CNN) or Histogram of Oriented Gradients (HOG)')
    parser.add_argument('-b', '--blurr', default=21, type=int, help='gaussian blurr kernel size (the neighbors to be considered), number must be odd')
    parser.add_argument('-u', '--upsampling', default=1, type=int, help='upsampling factor, higher values increase face recognition rate')
    parser.add_argument('-z', '--zoom_in', default=1, type=int, help='zoom in factor, higher values increase face detection speed')
    args = parser.parse_args()

    if args.blurr % 2 == 0:
        raise argparse.ArgumentTypeError("%s is not an odd int value" % args.blurr)

    if os.path.isfile(args.src):
        face_blur(args.src, args.dest, args)
    else:
        blur_all_photo(args.src, args.dest, args)
