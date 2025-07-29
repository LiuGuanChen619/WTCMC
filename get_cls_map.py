import numpy as np
import matplotlib.pyplot as plt
import torch


def get_classification_map(y_pred, y):

    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k]+1
                k += 1

    return  cls_labels

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([176, 47, 95]) / 255.
        if item == 2:
            y[index] = np.array([10, 254, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 4:
            y[index] = np.array([159, 32, 240]) / 255.
        if item == 5:
            y[index] = np.array([127, 255, 213]) / 255.
        if item == 6:
            y[index] = np.array([0, 139, 138]) / 255.
        if item == 7:
            y[index] = np.array([4, 205, 0]) / 255.
        if item == 8:
            y[index] = np.array([101, 174, 255]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 10:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 11:
            y[index] = np.array([215, 191, 216]) / 255.
        if item == 12:
            y[index] = np.array([254, 127, 80]) / 255.
        if item == 13:
            y[index] = np.array([160, 81, 45]) / 255.
        if item == 14:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 15:
            y[index] = np.array([218, 112, 214]) / 255.
        if item == 16:
            y[index] = np.array([0, 96, 255]) / 255.

        if item == 17:
            y[index] = np.array([128, 0, 128]) / 255.  # 紫色
        if item == 18:
            y[index] = np.array([255, 165, 0]) / 255.  # 橙色
        if item == 19:
            y[index] = np.array([0, 128, 0]) / 255.  # 绿色
        if item == 20:
            y[index] = np.array([0, 255, 255]) / 255.  # 青色
        if item == 21:
            y[index] = np.array([128, 128, 0]) / 255.  # 橄榄色
        if item == 22:
            y[index] = np.array([255, 192, 203]) / 255.  # 粉色

    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def Test(device, net, test_loader):
    net.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 避免计算图生成，节省显存
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            pred= net(data, label)
            pred_class = torch.argmax(pred, dim=1).cpu().numpy()
            label = label.cpu().numpy()

            all_preds.append(pred_class)
            all_labels.append(label)

    y_pred_test = np.concatenate(all_preds)
    y_test = np.concatenate(all_labels)
    return y_pred_test, y_test

def get_cls_map(net, device, all_data_loader, y):

    y_pred, y_new = Test(device, net, all_data_loader)
    cls_labels = get_classification_map(y_pred, y)
    x = np.ravel(cls_labels)
    gt = y.flatten()
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
    
    classification_map(y_re, y, 300, './maps/predictions.eps')
    classification_map(y_re, y, 300, './maps/predictions.png')
    classification_map(gt_re, y, 300,'./maps/gt.png')
    print('------Get classification maps successful-------')


