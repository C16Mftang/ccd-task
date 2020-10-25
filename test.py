import random

import psychopy.visual
import psychopy.event

win = psychopy.visual.Window(
    size=[400, 400],
    units="pix",
    fullscr=False
)

n_dots = 200

dot_xys = []

for dot in range(n_dots):

    dot_x = random.uniform(-200, 200)
    dot_y = random.uniform(-200, 200)

    dot_xys.append([dot_x, dot_y])

dot_stim = psychopy.visual.ElementArrayStim(
    win=win,
    units="pix",
    nElements=n_dots,
    elementTex=None,
    elementMask="circle",
    xys=dot_xys,
    sizes=10
)

dot_stim.draw()

win.flip()

psychopy.event.waitKeys()

win.close()
