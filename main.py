import cv2

def main():
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        cv2.imshow('pic', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()     

if __name__ == "__main__":
    main()