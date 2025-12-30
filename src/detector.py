import cv2 
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.5,track_con=0.5):
        """
        Initializes the HandDetector with specified parameters.
        
        :param mode: False for live video. True for static images.
        :param max_hands: How many hands to detect.
        :param detection_con: Detection confidence threshold (0.0 - 1.0).   
        :param track_con: Detection tracking confidence threshold (0.0 - 1.0).
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        # Initialize MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils #Drawing utilities

    def find_hands(self, img, draw=True):
        """
        Processes the image to find hands
        
        :param img: Actual Frame from the webcam
        :param draw: If True, draws hand landmarks on the image.
        :return: Processed image
        """

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Convert BGR to RGB
        self.results = self.hands.process(img_rgb) #We try to detect hands
        
        #If hands are detected
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    #Draw landmarks and connections
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return img
    
    def find_position(self,img,hand_no=0):
        """
        Obtain the position (x,y) of hand landmarks
        
        :return: List of landmark positions - [id, x, y]
        """
        lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]

            # Visit each landmark
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

        return lm_list
    