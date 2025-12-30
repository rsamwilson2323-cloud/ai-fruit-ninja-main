import cv2
import numpy as np
import random
import math
from src.utils import overlay_transparent

class Fruit:
    def __init__(self, screen_width, screen_height, image, is_bomb=False):
        """
        Initializes a Fruit object with random position and type.
        """

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.image = image
        self.is_bomb = is_bomb

        if self.image is not None:
            self.image = cv2.resize(self.image, (80, 80))

        # Initial position: Lower part of the screen with random x-coordinate
        self.x = random.randint(100, screen_width - 100)
        self.y = screen_height + 50  # Start off-screen

        # Physics / Velocity
        self.speed_x = random.randint(-10, 10)
        
        if self.is_bomb:
            self.speed_y = random.randint(-40, -32)
        else:
            self.speed_y = random.randint(-35, -28) # Upward velocity
            
        self.gravity = 1 # Gravity effect

    def update(self):
        """
        Updates the fruit's position based on its velocity and gravity.
        Returns True if the fruit is still on screen, False otherwise.
        """

        self.x += self.speed_x # Move in x
        self.y += self.speed_y # Apply gravity to y-velocity
        self.speed_y += self.gravity

        if self.y > self.screen_height + 100:
            return False # Fruit is off-screen
        return True
    
    def draw(self, img):
        """
        Draws the fruit on the given image using the PNG sprite.
        """
        if self.image is not None:
            overlay_transparent(img, self.image, int(self.x), int(self.y))

    def check_collision(self, finger_x, finger_y):
        """
        Calculate if the finger collides with the fruit.
        Returns True if collision occurs, False otherwise.
        """
        if (self.x < finger_x < self.x + 80) and (self.y < finger_y < self.y + 80):
            return True
        
        return False