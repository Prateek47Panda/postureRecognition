import cv2
import numpy as np

import api


def get_embed(img):
    return api.human_vector(img)[0]

#the euclidean distance between the two embeddings
def compare(emb1, emb2):
    return np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))


def get_cropped(img, y_min_a, x_min_a, y_max_a, x_max_a):
    height, width, channel = img.shape
    left = int(x_min_a * width)
    right = int(x_max_a * width)
    top = int(y_min_a * height)
    if top < 0:
        top = 0
    bottom = int(y_max_a * height)
    cropped = img[top:bottom, left:right]
    return cropped, img


def display(img):
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.imshow("test", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    name = ["x", "SAM", "SAMMY", "DANIEL", "ROBERT", "PTRICK", "CHRIS", "SONY", "IRFAN", "VIRAT", "EDGE", "ROCKI",
            "TAG"]
    cap = cv2.VideoCapture("asd.mp4")
    buffer = []
    average_minimum_d = 0
    # for people in sorted(os.listdir("buffer")):
    #     cluster = []
    #     for i, p in enumerate(sorted(os.listdir("buffer/" + people))):
    #         img = cv2.imread("buffer/" + people + "/" + p)
    #         emb = get_embed(img)
    #         cluster.append(emb)
    #     buffer.append(cluster)
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))
    min_thresh = 10.5
    while True:
        ret, image_np = cap.read()
        for_display = image_np.copy()
        for_draw = image_np.copy()
        for_display = cv2.cvtColor(for_display, cv2.COLOR_RGB2BGR)
        for_crop = for_display.copy()
        boxes = api.human_locations(for_display)
        assigned = []
        for box in boxes:
            leftbottom = box[0]
            righttop = box[1]
            left, bottom = leftbottom[0], leftbottom[1]
            right, top = righttop[0], righttop[1]
            frame = api.crop_human(for_crop, box)
            emb = get_embed(frame)
            mean_distances = []
            for cluster in buffer:
                distances = [compare(j, emb) for j in cluster]
                mean_distances.append(np.mean(distances))
            if len(mean_distances) > 0:
                min_d = min(mean_distances)
                index = mean_distances.index(min_d)
                if min_d <= min_thresh and index not in assigned:
                    assigned.append(index)
                    buffer[index].append(emb)
                    if len(buffer[index]) >= 100:
                        buffer[index].pop(0)
                    for_display = cv2.rectangle(for_draw, (left, top), (right, bottom),
                                                (255, 255, 255), 6)
                    for_display = cv2.putText(for_draw, name[index], (left, top),
                                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
                else:
                    if min_d - min_thresh >= 10:
                        buffer.append([emb])
                    average_minimum_d = min_d
                    # for_display = cv2.rectangle(for_display, (left, top), (right, bottom), (0, 0, 255), 2)
            if len(buffer) == 0:
                buffer.append([1])
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        for_save = for_draw.copy()
        if ret:
            out.write(for_save)
            cv2.imshow("test", for_draw)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            break
cap.release()
out.release()
cv2.destroyAllWindows()
