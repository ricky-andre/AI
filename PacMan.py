import pygame

# Define some colors
BLACK    = (   0,   0,   0)
WHITE    = ( 255, 255, 255)
GREEN    = (   0, 255,   0)
RED      = ( 255,   0,   0)
PATHS    = (204 ,155 , 120)
GRASSWALL= (69  , 201,  54)
Stone    = (204 , 204, 204)

pygame.init()

# Set the width and height of the screen [width, height]
size = (800, 500)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Darudernauts")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

pygame.mixer.music.load('darude.mp3')
pygame.mixer.music.set_endevent(pygame.constants.USEREVENT)
pygame.mixer.music.play()

#player_image = pygame.image.load("C:\Users\Mark\Documents\player.gif").convert()
#shroom_image = pygame.image.load("C:\Users\Mark\Documents\shroom.gif").convert()

# Hide the mouse cursor
pygame.mouse.set_visible(0)
# Speed in pixels per frame
x_speed = 0
y_speed = 0
x = 350
y = 200

shroom_list_x = [150,150,150,150,150,150,150,150,150,150,600,600,600,600,600,600,600,600,600,600,200,250,300,350,400,450,500,550,  0, 50,100,650,700,750,  0, 50,100,650,700,750,200,250,300,350,400,450,500,550,300,350,400,450,250,250,250,250,250,250,500,500,500,500,500,500,350,400,350,400,400,-50]
shroom_list_y = [  0, 50,100,150,200,250,300,350,400,450,  0, 50,100,150,200,250,300,350,400,450, 50, 50, 50, 50, 50, 50, 50, 50,100,100,100,100,100,100,350,350,350,350,350,350,400,400,400,400,400,400,400,400,300,300,300,300,100,150,200,250,300,350,100,150,200,250,300,350,100,100,150,150,200,-50]

score = 0
# -------- Main Program Loop -----------
while not done:
    # --- Main event loop
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
            done = True # Flag that we are done so we exit this loop

    move_horizontally = 0
    move_vertically = 0

    if event.type == pygame.KEYDOWN:
        # Figure out if it was an arrow key. If so
        # adjust speed.
        if event.key == pygame.K_LEFT:
            move_horizontally = -50
        if event.key == pygame.K_RIGHT:
            move_horizontally = 50
        if event.key == pygame.K_UP:
            move_vertically = -50
        if event.key == pygame.K_DOWN:
            move_vertically = 50
    # --- Game logic should go here
    #side tps
    if x < -50:
        x = 750
    if x > 800:
        x = -50
    #top/bot tps
    if y < -50:
        y = 450
    if y > 500:
        y = -50

    #midtop
    if x == 150 and y == 0:
        move_horizontally = 0#left

    if x == 600 and y == 0:
        move_horizontally = 0#right

    if x == 150 and y == 450:
        move_horizontally = 0#left

    if x == 600 and y == 450:
        move_horizontally = 0#right

    if x == 150 and y == 50:
        if move_horizontally < 0:
            move_horizontally = 0

    if x == 600 and y == 50:
        if move_horizontally > 0:
            move_horizontally = 0

    if x == 600 and y == 400:
        if move_horizontally > 0:
            move_horizontally = 0

    if x >= 200 and x <= 550 and y == 50:
        if move_vertically < 0:
            move_vertically = 0


    if x == 200 or x == 300 or x == 450 or x == 550:
        if y == 50:
            move_vertically = 0

    if x == 150:
        if y >= 0 and y <= 450:
            if y != 100 and y != 350:
                if move_horizontally < 0:
                    move_horizontally =0

    if x == 200 and y == 400:
        move_vertically = 0

    if x == 550 and y == 400:
        move_vertically = 0

    if x == 300 or x == 350 or x == 400 or x == 450:
        if y == 400:
            move_vertically = 0


    #topleft path
    if x >= 0 and x <= 100 and y == 100:
        move_vertically = 0
    #botright path
    if x >= 0 and x <= 100 and y == 350:
        move_vertically = 0
    #topright path
    if x >= 650 and x <= 800 and y == 100:
        move_vertically = 0
    #botright path
    if x >= 650 and x <= 800 and y == 350:
        move_vertically = 0

    if x == 150:
        if y == 150 or y == 200 or y == 250 or y == 300:
            move_horizontally = 0

    if x == 600:
        if y == 150 or y == 200 or y == 250 or y == 300:
            move_horizontally = 0

    if y == 100 and x == 150:
        if move_horizontally >= 0:
            move_horizontally = 0

    if y == 350 and x == 150:
        if move_horizontally >= 0:
            move_horizontally = 0

    if y == 100 and x == 600:
        if move_horizontally <= 0:
            move_horizontally = 0

    if y == 350 and x == 600:
        if move_horizontally <= 0:
            move_horizontally = 0

    if x == 250 and y == 400:
        if move_vertically > 0:
            move_vertically = 0

    if x == 500 and y == 400:
        if move_vertically > 0:
            move_vertically = 0

    if x == 250 and y == 350:
        move_horizontally = 0

    if x == 500 and y == 350:
        move_horizontally = 0

    if x < 0 or x > 800:
        move_vertically = 0

    if y < 0 or y == 500:
        move_horizontally = 0

    if y == 300:
        if x == 300 or x == 350 or x == 400 or x == 450:
            move_vertically = 0

    if y == 300 and x == 250:
        if move_horizontally < 0:
            move_horizontally = 0

    if y == 300 and x == 500:
        if move_horizontally > 0:
            move_horizontally = 0

    if x == 250 or x == 500:
        if y == 100 or y == 150 or y == 200 or y == 250:
            move_horizontally = 0

    if x == 350:
        if y == 100 or y == 150:
            if move_horizontally < 0:
                move_horizontally = 0

    if x == 400:
        if y == 100 or y == 150:
            if move_horizontally > 0:
                move_horizontally = 0

    if x == 350 and y == 200:
        if move_horizontally < 0:
            move_horizontally = 0
        elif move_vertically > 0:
            move_vertically = 0

    if x == 400 and y == 200:
        if move_horizontally > 0:
            move_horizontally = 0
        elif move_vertically > 0:
            move_vertically = 0

    x = x + move_horizontally
    y = y + move_vertically


    if score == 69:
        print ("VICTORY!!! YOU'VE COLLECTED 69 MUSHROOMS!!!")

    # --- Drawing code should go here

    # First, clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.
    screen.fill(Stone)

    pygame.draw.rect(screen,GRASSWALL,[0,0,150,100])#TOPLEFT
    pygame.draw.rect(screen,GRASSWALL,[0,150,150,200])#MIDLEFT
    pygame.draw.rect(screen,GRASSWALL,[0,400,150,100])#BOTLEFT

    pygame.draw.rect(screen,GRASSWALL,[650,0,150,100])#TOPRIGHT
    pygame.draw.rect(screen,GRASSWALL,[650,150,150,200])#MIDRIGHT
    pygame.draw.rect(screen,GRASSWALL,[650,400,150,100])#BOTRIGHT

    pygame.draw.rect(screen,GRASSWALL,[200,0,400,50])#MIDTOP
    pygame.draw.rect(screen,GRASSWALL,[200,450,400,50])#MIDBOT

    pygame.draw.rect(screen,GRASSWALL,[200,100,50,300])#JUNGLEFT
    pygame.draw.rect(screen,GRASSWALL,[550,100,50,300])#JUNGRIGHT
    pygame.draw.rect(screen,GRASSWALL,[300,350,200,50])#JUNGBOT

    pygame.draw.rect(screen,GRASSWALL,[300,250,200,50])#BASEBOT
    pygame.draw.rect(screen,GRASSWALL,[300,100,50,150])#BASELEFT
    pygame.draw.rect(screen,GRASSWALL,[450,100,50,150])#BASERIGHT

#    player_image = pygame.image.load("C:\Users\Mark\Documents\player.gif").convert()
#    shroom_image = pygame.image.load("C:\Users\Mark\Documents\shroom.gif").convert()
    # --- Go ahead and update the screen with what we've drawn.
        # Get the current mouse position. This returns the position
    # as a list of two numbers.

    x += x_speed
    y += y_speed

##    print x
##    print y

    # Copy image to screen:
    screen.blit(player_image, [x, y])

    for shrm in range(0, len(shroom_list_x)-1):
        if x == shroom_list_x[shrm] and y == shroom_list_y[shrm]:
            shroom_list_x.pop(shrm)
            shroom_list_y.pop(shrm)
            score += 1
            print ("SCORE:",score,"mushrooms")

        screen.blit(shroom_image, [shroom_list_x[shrm], shroom_list_y[shrm]])
    pygame.display.flip()

    # --- Limit to 7 frames per second
    clock.tick(10)

# Close the window and quit.
# If you forget this line, the program will 'hang'
# on exit if running from IDLE.
pygame.quit()