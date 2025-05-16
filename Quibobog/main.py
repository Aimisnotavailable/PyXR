import cv2
import mediapipe as mp
# import pyautogui as pg
import math
import pygame
import sys
import random
import hashlib

from scripts.assets import Assets
from scripts.camera import Follow
from scripts.engine import Engine
from scripts.utils import Animation, load_image, load_sound, load_sounds
from scripts.logger import get_logger_info
from scripts.sparks import Sparks

SCALE_IDX = (9, 0)
L_CLICK_IDX = 8

MAIN_CONTROL_IDX = 4

SENSITIVITY = 3

SCALE = 0.5
MAX_SCALE = SCALE * 2
MIN_SCALE = SCALE // 2

# MONOLOGUE_COOLDOWN = 20
ATTACK_COOLDOWN = 20
HISTOGRAM_SIZE = 5
VELOCITY_IDX = 9
SPARK_COUNT = 20
PUSH_FORCE = 20
QUIBOCOINCOUNT = 0
COST = 5

def get_dist(start_pos, end_pos, dir=0):
    if dir:
        return [-(start_pos[0] - end_pos[0]) / HISTOGRAM_SIZE, -(start_pos[1] - end_pos[1]) / HISTOGRAM_SIZE]
    return math.sqrt(max(0, (end_pos[0] - start_pos[0]) + (end_pos[1] - start_pos[1])))

def get_scale(start_pos, end_pos):
    return get_dist(start_pos, end_pos) / SCALE

class Game(Engine):

    def __init__(self, dim=..., font_size=20):
        super().__init__(dim, font_size)

        pygame.mixer.init()
        self.camera = Follow()
        self.assetss = Assets()

        self.cap = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
                        max_num_hands=2, 
                        min_detection_confidence=0.2,  # Corrected argument name
                        min_tracking_confidence=0.2
                    )

        self.position_histogram = []

        self.block = WorldObj(self, img=load_image('obj\\0.png', [125, 125]), 
                              pos = [self.display.get_width() // 2, 
                                     self.display.get_height() // 2])
        
        self.sounds = {'coin' : load_sound('coin.wav'),
                       'hurt' : load_sound('hurt.wav'),
                       'chaching' : load_sound('chaching.wav'),
                       'monologue' : load_sounds('monologue')}
        
        self.sparks : list[Sparks] = []
        self.quibocoincount = 0
        self.multiplier = 1

    def render_hands(self, positions):
        # get_logger_info('APP', f'{type(positions.landmark)}')
        render_positions = []

        for position in positions.landmark:
            render_positions.append([ -position.x * self.display.get_width() + self.display.get_width(), position.y * self.display.get_height()])
            pygame.draw.circle(self.display, 
                               (255, 255, 255), 
                               render_positions[-1],
                                2)
            
        
            if not self.block.collidercd and self.block.rect().collidepoint(render_positions[-1]) and len(self.position_histogram) == HISTOGRAM_SIZE:
                vel = self.calculate_velocity(dir=1)
                self.block.velocity = [random.random() + PUSH_FORCE if vel[0] > 1 else random.random() - PUSH_FORCE + 1, random.random() + PUSH_FORCE if vel[1] > 1 else random.random() - PUSH_FORCE + 1]
                self.block.collidercd = ATTACK_COOLDOWN
                
                self.sounds['hurt'].play()

                if random.randint(0, 1):
                    self.sounds['monologue'][random.randint(0, len(self.sounds['monologue']) - 1)].play()

                if random.random() < (0.2 * self.multiplier):
                    self.quibocoincount += 1
                    self.sounds['coin'].play()

                for i in range(SPARK_COUNT):
                    self.sparks.append(Sparks(random.random() * math.pi * 2, random.random() + 3, render_positions[-1].copy(), (255, 0, 0)))

        if len(self.position_histogram) < HISTOGRAM_SIZE:
            self.position_histogram.append(render_positions[VELOCITY_IDX])
        else:
            self.position_histogram.pop(0)
            self.position_histogram.append(render_positions[VELOCITY_IDX])

        for conn in self.mp_hands.HAND_CONNECTIONS:
            pygame.draw.line(self.display, (0, 0, 255), render_positions[conn[0]], render_positions[conn[1]])

    def calculate_velocity(self, dir=0):
        if len(self.position_histogram) == HISTOGRAM_SIZE:
            return get_dist(self.position_histogram[0], self.position_histogram[-1], dir=dir)
        return [0, 0]

    def run(self):
        
        while True:
            
            self.display.fill((0, 0, 0))

            ret, frame = self.cap.read()

            if not ret:
                break
            
            # Convert frame to RGB (required by MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand tracking
            results = self.hands.process(rgb_frame)
            # Draw landmarks if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.render_hands(hand_landmarks)
            #         mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # #print(hand_landmarks)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_u and self.quibocoincount >= COST * self.multiplier:
                        self.quibocoincount -= COST * self.multiplier
                        self.multiplier += 1
                        self.sounds['chaching'].play()

            for spark in self.sparks:
                spark.render(self.display)

                if spark.update():
                    self.sparks.remove(spark)

            self.block.update(self.display)
            self.display.blit(self.font.render(f'Wallet : {hashlib.md5('thsisawallet'.encode()).hexdigest()}', True, (255, 255, 255)), (0, 0))
            self.display.blit(self.font.render(f'QuiboCoin: {self.quibocoincount}', True, (255, 255, 255)), (0, 20))
            self.display.blit(self.font.render(f'Next ugprade cost : {self.multiplier * COST}', True, (255, 255, 255)), (0, self.display.get_height() - 40))
            self.display.blit(self.font.render(f'Multiplier: {self.multiplier}', True, (255, 255, 255)), (0, self.display.get_height() - 20))
            
            get_logger_info('CORE', f'{self.calculate_velocity(dir=1)}')

            self.screen.blit(pygame.transform.scale(self.display, self.screen.get_size()), (0, 0))

            pygame.display.update()
            self.clock.tick()

class WorldObj:

    def __init__(self, game : Game, img = None, pos = [0, 0]):
        self.game = game
        self.img = img
        self.pos = list(pos)
        self.velocity = [0, 0]
        self.collidercd = 0
        self.angle = 0
    
    def rect(self):
        return pygame.Rect(*self.pos, *self.img.get_size())
    
    def generate_sparks(self, pos):
        for i in range(SPARK_COUNT):
            self.game.sparks.append(Sparks(random.random() * math.pi * 2, random.random() + 3, pos, (255, 0, 0)))

    def update(self, render_surf, offset=[0, 0]):

        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]
        
        if int(self.pos[0]) <= 0:
            self.pos[0] = 0
            self.velocity[0] *= -1
            self.generate_sparks(self.pos.copy())

        elif int(self.pos[0] + self.rect()[2]) >= render_surf.get_width():
            self.pos[0] = render_surf.get_width() - self.rect()[2]
            self.velocity[0] *= -1
            self.generate_sparks([self.pos[0] + render_surf.get_width(), self.pos[1]])
        
        if int(self.pos[1]) <= 0:
            self.pos[1] = 0
            self.velocity[1] *= -1
            self.generate_sparks(self.pos.copy())

        elif int(self.pos[1] + self.rect()[3]) >= render_surf.get_height():
            self.pos[1] = render_surf.get_height() - self.rect()[3]
            self.velocity[1] *= -1
            self.generate_sparks([self.pos[0], self.pos[1] + self.rect()[3]])

        if self.velocity[0] <= 0:
            self.velocity[0] = min(0, self.velocity[0] + 0.2)
        elif self.velocity[0] >= 0:
            self.velocity[0] = max(0, self.velocity[0] - 0.2)

        if self.velocity[1] <= 0:
            self.velocity[1] = min(0, self.velocity[1] + 0.2)
        elif self.velocity[1] >= 0:
            self.velocity[1] = max(0, self.velocity[1] - 0.2)

        self.render(render_surf, offset=offset)

        self.collidercd = max(0, self.collidercd-1)
    
    def render(self, surf : pygame.Surface, offset=[0, 0]):
        if not self.velocity[0]:
            surf.blit(self.img, (self.pos[0] - offset[0], self.pos[1] - offset[1]))
        else:
            angle = self.angle #math.radians(self.angle)
            img = pygame.transform.rotate(self.img, angle)
            img_rect = img.get_rect(center=(self.pos[0] + math.cos(math.radians(angle)), self.pos[1] + math.sin(math.radians(angle))))
            surf.blit(img, img_rect)
            self.angle += random.random() + 20


Game((1600, 900)).run()