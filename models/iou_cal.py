import torch

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (x0, y0, x1, y1), which reflects
            (bottom, left, top, right)
    :param rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    bottom_line = max(rec1[1], rec2[1])
    top_line = min(rec1[3], rec2[3])

    # judge if there is an intersect
    if left_line >= right_line or bottom_line >= top_line:
        return 0
    else:
        intersect = (right_line - left_line) * (top_line - bottom_line)
        return intersect / (sum_area - intersect)

def build_graph(box_dets):
    """
    box_dets: torch.Tensor(T, n, 5)
              the coordinate of faster-rcnn, (x, y, x+w, y+h)
    return g_front, g_back: torch.Tensor(N, N)
    """
    frame_num, bbox_num_pre_frame, _ = box_dets.shape
    bboxes_num = bbox_num_pre_frame * frame_num
    g_front = torch.zeros(bboxes_num, bboxes_num)
    g_back = torch.zeros(bboxes_num, bboxes_num)
    # print(g_front[0, :])
    # g_front
    for i in range(bboxes_num):
        for j in range(bboxes_num):
            if (i//bbox_num_pre_frame >= j//bbox_num_pre_frame) or (j//bbox_num_pre_frame - i//bbox_num_pre_frame) > 1:
                g_front[i, j] = 0
            else:
                rect1 = box_dets[i//bbox_num_pre_frame, i%bbox_num_pre_frame, 1:5]
                rect2 = box_dets[j//bbox_num_pre_frame, j%bbox_num_pre_frame, 1:5]
                g_front[i, j] = compute_iou(rect1, rect2)

    # print(g_front[0, :])

    # g_back
    for i in range(bboxes_num):
        for j in range(bboxes_num):
            if (j//bbox_num_pre_frame >= i//bbox_num_pre_frame) or (i//bbox_num_pre_frame - j//bbox_num_pre_frame) > 1:
                g_back[i, j] = 0
            else:
                rect1 = box_dets[i//bbox_num_pre_frame, i%bbox_num_pre_frame, 1:5]
                rect2 = box_dets[j//bbox_num_pre_frame, j%bbox_num_pre_frame, 1:5]
                g_back[i, j] = compute_iou(rect1, rect2)

    # print(g_back[0, :])
    return g_front, g_back


if __name__ == '__main__':
    box_dets = torch.rand(4, 50, 5)
    print(build_graph(box_dets))