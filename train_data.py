import numpy as np

# Input dataset
train_data = np.array([
    [25, 30, 50, 400, 0],
[32, 20, 100, 600, 1],
[28, 40, 200, 350, 0],
[29, 35, 180, 450, 0],
[31, 25, 250, 520, 1],
[27, 50, 300, 490, 0],
[35, 30, 350, 300, 1],
[30, 40, 400, 700, 0],
[33, 20, 120, 380, 1],
[26, 45, 90, 650, 0],
[34, 25, 310, 410, 1],
[28, 38, 230, 500, 0],
[31, 33, 450, 200, 1],
[29, 29, 100, 300, 0],
[30, 42, 120, 450, 1],
[32, 28, 280, 480, 0],
[36, 30, 360, 520, 1],
[25, 40, 220, 650, 0],
[29, 31, 170, 450, 1],
[33, 35, 320, 400, 0],
[34, 25, 180, 380, 1],
[30, 30, 260, 490, 0],
[35, 28, 200, 350, 1],
[29, 42, 300, 400, 0],
[32, 25, 240, 500, 1],
[28, 35, 320, 450, 0],
[27, 30, 180, 350, 1],
[34, 40, 290, 480, 0],
[36, 25, 310, 500, 1],
[30, 38, 360, 520, 0],
[32, 33, 250, 450, 1],
[29, 40, 400, 300, 0],
[28, 28, 220, 420, 1],
[27, 50, 340, 470, 0],
[33, 25, 120, 380, 1],
[31, 40, 270, 600, 0],
[36, 28, 300, 450, 1],
[29, 33, 260, 400, 0],
[34, 42, 320, 480, 1],
[30, 28, 280, 350, 0],
[25, 50, 240, 500, 1],
[28, 35, 300, 600, 0],
[32, 25, 180, 350, 1],
[33, 42, 360, 450, 0],
[31, 28, 310, 480, 1],
[29, 38, 200, 470, 0],
[30, 30, 270, 650, 1],
[34, 33, 250, 400, 0],
[28, 40, 290, 380, 1],
[36, 35, 310, 500, 0]
])

# Output dataset
train_labels = np.array([
    [50, 60, 70, 80, 0],
[80, 90, 100, 70, 1],
[60, 50, 40, 90, 0],
[70, 60, 80, 80, 0],
[90, 70, 100, 60, 1],
[50, 80, 60, 70, 0],
[100, 60, 90, 80, 1],
[70, 50, 100, 90, 0],
[80, 40, 70, 60, 1],
[60, 90, 50, 80, 0],
[90, 70, 80, 60, 1],
[70, 60, 90, 50, 0],
[100, 50, 100, 80, 1],
[50, 70, 70, 60, 0],
[80, 60, 60, 90, 1],
[90, 50, 80, 80, 0],
[100, 60, 90, 70, 1],
[70, 50, 100, 60, 0],
[60, 70, 60, 80, 1],
[80, 60, 90, 50, 0],
[90, 50, 80, 80, 1],
[70, 60, 60, 70, 0],
[100, 50, 70, 90, 1],
[50, 80, 90, 60, 0],
[80, 50, 80, 70, 1],
[60, 70, 100, 90, 0],
[50, 60, 70, 80, 1],
[90, 80, 60, 70, 0],
[100, 70, 50, 80, 1],
[60, 90, 70, 60, 0],
[80, 60, 90, 70, 1],
[70, 50, 80, 80, 0],
[50, 60, 100, 70, 1],
[80, 90, 90, 60, 0],
[90, 50, 80, 80, 1],
[70, 60, 100, 70, 0],
[100, 50, 60, 90, 1],
[50, 70, 80, 60, 0],
[60, 60, 90, 50, 1],
[80, 90, 70, 80, 0],
[90, 70, 50, 70, 1],
[70, 50, 60, 90, 0],
[50, 60, 100, 70, 1],
[80, 90, 80, 60, 0],
[90, 50, 60, 80, 1],
[60, 70, 90, 70, 0],
[50, 60, 70, 90, 1],
[80, 50, 60, 80, 0],
[90, 70, 50, 70, 1],
[60, 80, 100, 60, 0]
])