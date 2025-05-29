import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("برخورد دو میوه")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)

# Ball properties
class Ball:
    def __init__(self, x, y, radius, color, speed_x, speed_y):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.speed_x = speed_x
        self.speed_y = speed_y

    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y

        # Bounce off walls
        if self.x - self.radius < 0 or self.x + self.radius > WIDTH:
            self.speed_x *= -1
        if self.y - self.radius < 0 or self.y +:
            self.speed_y *= -1

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def check_collision(self, other):
        distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        if distance <= self.radius + other.radius:
            self.speed_x, other.speed_x = other.speed_x, self.speed_x
            self.speed_y, other.speed_y = other.speed_y, self.speed_y

# Create balls (representing two fruits)
fruit1 = Ball(random.randint(50, WIDTH-50), random.randint(50, HEIGHT-50), 30, RED, 3, 4)
fruit2 = Ball(random.randint(50, WIDTH-50), random.randint(50, HEIGHT-50), 30, ORANGE, -3, -4)

# Main loop
running = True
clock = pygame.time.Clock()
while running:
    screen.fill(WHITE)
    
    fruit1.move()
    fruit2.move()
    fruit1.check_collision(fruit2)
    
    fruit1.draw()
    fruit2.draw()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
