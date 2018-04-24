import cv2
import numpy as np
import math
import random
import time

size = 200
h = size
w = size
r = size/2


def get_color_clock(hours, minutes):
    img = np.zeros((h, w, 3), np.uint8)
    img.fill(255)

    cv2.circle(img, (int(w / 2), int(h / 2)), r, (0, 0, 0), 10)

    value = hours * 60 + minutes

    deg_hour = 2 * math.pi / 12
    deg_min = 2 * math.pi / 60

    hours_angle = hours * deg_hour + (deg_hour * minutes / 60)
    min_angle = minutes * deg_min

    hours_angle -= math.pi / 2
    min_angle -= math.pi / 2

    h_x = 0.5 * r * math.cos(hours_angle) + w / 2
    h_y = 0.5 * r * math.sin(hours_angle) + h / 2

    m_x = 0.9 * r * math.cos(min_angle) + w / 2
    m_y = 0.9 * r * math.sin(min_angle) + h / 2

    cv2.line(img, (int(w / 2), int(h / 2)), (int(h_x), int(h_y)), (255, 0, 0), 10)
    cv2.line(img, (int(w / 2), int(h / 2)), (int(m_x), int(m_y)), (0, 255, 0), 5)

    return img, value


def draw_clock(hours, minutes):
    img = np.zeros((h, w), np.uint8)
    img.fill(255)

    cv2.circle(img, (int(w / 2), int(h / 2)), int(r), (0, 0, 0), 10)

    for i in range(12):
        angle = 2 * math.pi / 12 * i
        x1 = r * math.cos(angle) + w / 2
        y1 = r * math.sin(angle) + h / 2
        x2 = 0.85 * r * math.cos(angle) + w / 2
        y2 = 0.85 * r * math.sin(angle) + h / 2

        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 3)

    for i in range(60):
        angle = 2 * math.pi / 60  * i
        x1 = r * math.cos(angle) + w / 2
        y1 = r * math.sin(angle) + h / 2
        x2 = 0.9 * r * math.cos(angle) + w / 2
        y2 = 0.9 * r * math.sin(angle) + h / 2

        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 3)

    value = hours * 60 + minutes

    deg_hour = 2 * math.pi / 12
    deg_min = 2 * math.pi / 60

    hours_angle = hours * deg_hour + (deg_hour * minutes / 60)
    min_angle = minutes * deg_min

    hours_angle -= math.pi / 2
    min_angle -= math.pi / 2

    h_x = 0.5 * r * math.cos(hours_angle) + w / 2
    h_y = 0.5 * r * math.sin(hours_angle) + h / 2

    m_x = 0.9 * r * math.cos(min_angle) + w / 2
    m_y = 0.9 * r * math.sin(min_angle) + h / 2

    cv2.line(img, (int(w / 2), int(h / 2)), (int(h_x), int(h_y)), (0, 0, 0), 15)
    cv2.line(img, (int(w / 2), int(h / 2)), (int(m_x), int(m_y)), (0, 0, 0), 7)

    return img, value


def get_random_clock(color = False):

    #hours = random.choice([0,3,6,9])
    hours = random.randint(0, 11)

    #minutes = random.randint(0,59)
    minutes = random.choice([0, 15, 30, 45])
    #minutes = 0

    if(color):
        img, value = get_color_clock(hours, minutes)
    else:
        img, value = draw_clock(hours, minutes)

    return img, np.array([value])


def get_random_clock_batch(batch_size, color=False):

    batch_x = []
    batch_y = []

    for _ in range(batch_size):

        img, value = get_random_clock(color)

        batch_x.append(img)
        batch_y.append(value)

    return np.array(batch_x), np.array(batch_y)


def value_to_time(value):
    min = int(value % 60)
    h = int((value-min)/60)

    return h, min


def value_to_string(value):
    h, min = value_to_time(value)
    return str(h) + ":" + format(min, '02')


def random_test():
    img, value = get_random_clock()
    print(img.shape)
    print(value)

    print(value_to_string(value))
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_clock():
    for i in range(720):
        h, m = value_to_time(i)

        img, _ = draw_clock(h ,m)
        cv2.imshow('image', img)
        time.sleep(.01)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    #random_test()
    run_clock()