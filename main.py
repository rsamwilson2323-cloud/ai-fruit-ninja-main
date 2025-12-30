import cv2
import time 
import numpy as np
import random
import os
from collections import deque
from src.detector import HandDetector
from src.game import Fruit

def main():
    # Configure webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    WIDTH, HEIGHT = 1280, 720
    cap.set(3, WIDTH) # Width
    cap.set(4, HEIGHT) # Height

    # --- LOAD ASSETS (SPRITES) ---
    # Loading images from the assets folder
    img_sandia = cv2.imread("assets/sandia.png", cv2.IMREAD_UNCHANGED)
    img_naranja = cv2.imread("assets/naranja.png", cv2.IMREAD_UNCHANGED)
    img_banana = cv2.imread("assets/banana.png", cv2.IMREAD_UNCHANGED) 
    img_bomb = cv2.imread("assets/bomb.png", cv2.IMREAD_UNCHANGED)

    # Basic check to ensure images are loaded
    if img_sandia is None or img_bomb is None:
        print("Error: Images not found in 'assets' folder.")
        return

    score = 0
    strikes = 0 # Lives lost
    game_over = False
    sword_points = deque(maxlen=10)

    # Detector
    detector = HandDetector(max_hands=1, detection_con=0.3, track_con=0.5)

    # VARIABLES OF THE GAME
    fruits = []
    last_spawn_time = time.time()
    spawn_interval = 0.9

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    print("Game Init. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        # Get actual frame size to handle different camera resolutions
        real_h, real_w, _ = frame.shape

        if game_over:
            overlay = frame.copy()
            # Use real_w and real_h to cover the whole screen
            cv2.rectangle(overlay, (0, 0), (real_w, real_h), (20, 20, 20), -1) 
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            
            font_title = cv2.FONT_HERSHEY_TRIPLEX
            font_text = cv2.FONT_HERSHEY_SIMPLEX
            
            title_text = "GAME OVER"
            title_scale = 3.5 
            title_thickness = 5
            
            (text_w, text_h), _ = cv2.getTextSize(title_text, font_title, title_scale, title_thickness)
            
            # Center based on actual frame width (real_w)
            pos_x_title = (real_w - text_w) // 2
            pos_y_title = (real_h // 2) - 80 
            
            cv2.putText(frame, title_text, (pos_x_title, pos_y_title), 
                        font_title, title_scale, (0, 0, 0), title_thickness + 8)
            cv2.putText(frame, title_text, (pos_x_title, pos_y_title), 
                        font_title, title_scale, (0, 0, 255), title_thickness)

            score_text = f"Final Score: {score}"
            (score_w, _), _ = cv2.getTextSize(score_text, font_text, 1.2, 2)
            # Center score using real_w
            cv2.putText(frame, score_text, ((real_w - score_w)//2, pos_y_title + 100), 
                        font_text, 1.2, (255, 255, 255), 2)
            
            instr_color = (200, 200, 200)
            instr_scale = 0.9
            
            instr1 = "Press 'r' to Restart"
            (i1_w, _), _ = cv2.getTextSize(instr1, font_text, instr_scale, 2)
            # Center instruction 1 using real_w 
            cv2.putText(frame, instr1, ((real_w - i1_w)//2, pos_y_title + 170), 
                        font_text, instr_scale, instr_color, 2)

            instr2 = "Press 'q' to Quit"
            (i2_w, _), _ = cv2.getTextSize(instr2, font_text, instr_scale, 2)
            # Center instruction 2 using real_w 
            cv2.putText(frame, instr2, ((real_w - i2_w)//2, pos_y_title + 210), 
                        font_text, instr_scale, instr_color, 2)
            
            cv2.imshow('AI Fruit Ninja - Dev Mode', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('r'): 
                score = 0
                strikes = 0
                fruits = []
                game_over = False
                sword_points.clear()
            continue


        # Detect hands and get landmark positions
        frame = detector.find_hands(frame, draw=False)
        lm_list = detector.find_position(frame)

        index_x, index_y = None, None

        if len(lm_list) != 0:
            index_x, index_y = lm_list[8][1], lm_list[8][2]
            sword_points.appendleft((index_x, index_y))
        else:
            if len(sword_points) > 0:
                sword_points.clear()

        
        for i in range(1, len(sword_points)):
            if sword_points[i-1] is None or sword_points[i] is None:
                continue
            
            factor = len(sword_points) - i
            thickness = int(np.sqrt(20 * float(factor)) / 1.5)

            if thickness < 1: thickness = 1

            cv2.line(frame, sword_points[i-1], sword_points[i], (255, 255, 255), thickness)


        # Manage fruit spawning & Difficulty
        current_time = time.time()
        
        # Increase difficulty: reduce interval based on score (min 0.3s)
        dynamic_interval = max(0.3, spawn_interval - (score // 10) * 0.05)

        if current_time - last_spawn_time > dynamic_interval:
            if random.random() < 0.25:
                # Spawn BOMB
                fruits.append(Fruit(screen_width=WIDTH, screen_height=HEIGHT, image=img_bomb, is_bomb=True))
            else:
                # Spawn FRUIT (Random choice)
                img_fruit = random.choice([img_sandia, img_naranja, img_banana])
                fruits.append(Fruit(screen_width=WIDTH, screen_height=HEIGHT, image=img_fruit, is_bomb=False))

            last_spawn_time = current_time

        # Collision logic and fruit updates
        for fruit in fruits[:]: 
            fruit.draw(frame)
            active = fruit.update()
            
            # 1. Collision Check
            if index_x is not None and index_y is not None:
                if fruit.check_collision(index_x, index_y):
                    if fruit.is_bomb:
                        game_over = True
                    else:
                        score += 1
                        fruits.remove(fruit) # Erase fruit on collision
                    continue 

            # 2. Logic for "Missed" fruits (Strikes)
            if not active:
                # If fruit goes off-screen and it wasn't a bomb, lose a life
                if not fruit.is_bomb:
                    strikes += 1
                    if strikes >= 3:
                        game_over = True
                
                fruits.remove(fruit)

        # UI Drawing
        cv2.rectangle(frame, (0, 0), (300, 110), (0, 0, 0), -1)
        
        # Score
        cv2.putText(frame, f'Score: {score}', (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
        
        # Strikes (Visual X X X)
        strikes_text = "X " * strikes
        cv2.putText(frame, f'Missed: {strikes_text}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Debug / Info (Optional)
        # cv2.putText(frame, f'Active: {len(fruits)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('AI Fruit Ninja - Dev Mode', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()