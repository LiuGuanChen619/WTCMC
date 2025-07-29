import os
import random
from distutils import dist
import h5py
import tifffile
import spectral

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv


import model


import time
import get_cls_map


dataset='IP'



Epoech = 100
Lr = 0.001


test_ratio = 0.95
val_ratio = 1-(5/95)

patch_size = 13
num_classes =16

batch_size = 64

pca_components = 90




def compute_class_weights(y_train):
    y_train = y_train.astype(np.int64)
    class_counts = np.bincount(y_train)

    class_counts[class_counts == 0] = 1

    class_weights = 1.0 / class_counts
    class_weights = class_weights * len(class_counts) / class_weights.sum()

    return torch.tensor(class_weights, dtype=torch.float32)




def loadData(name):

    data_path = os.path.join(os.getcwd(), '')

    if name == 'PU':
        data = sio.loadmat(os.path.join(data_path, './data/PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, './data/PaviaU_gt.mat'))['paviaU_gt']

    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, './data/Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, './data/Indian_pines_gt.mat'))['indian_pines_gt']
    if name == 'BS':
        data = sio.loadmat(os.path.join(data_path, './data/Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(data_path, './data/Botswana_gt.mat'))['Botswana_gt']




    return data, labels


def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))


    pca = PCA(n_components=numComponents, whiten=False)
    newX = pca.fit_transform(newX)


    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX



def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]),dtype=np.float32)


    x_offset = margin
    y_offset = margin

    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX



def createImageCubes(X, y, windowSize=13, removeZeroLabels=True):

    margin = int((windowSize - 1) / 2)

    X = X.astype(np.float32)
    y = y.astype(np.float32)


    zeroPaddedX = padWithZeros(X, margin=margin)


    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]),dtype=np.float32)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]),dtype=np.float32)


    patchIndex = 0


    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):

            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]


            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]


            patchIndex = patchIndex + 1


    if removeZeroLabels:

        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]


        patchesLabels -= 1

    return patchesData, patchesLabels



def splitTrainTestSet(X, y, testRatio, randomState):

    X_train, X_test_all, y_train, y_test_all = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)

    return X_train, X_test_all, y_train, y_test_all


def splitTrainTestSet_val(X, y, testRatio, randomState):

    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)

    return X_val, X_test, y_val, y_test


def create_data_loader(X, y, patch_size,random_seed , generator):

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA transformation ... ...')

    X_pca = applyPCA(X, numComponents=pca_components)


    print('Data shape after PCA: ', X_pca.shape)


    print('\n... ... create data cubes ... ...')

    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('删除标签为0后的数据 ', X_pca.shape)
    print('删除标签为0后的标签 ', y_all.shape)


    print('\n... ... 创建训练集和测试集 ... ...')


    X_train, X_test_all, y_train, y_test_all = splitTrainTestSet(X_pca, y_all, test_ratio, randomState=random_seed)
    X_val, X_test, y_val, y_test = splitTrainTestSet_val(X_test_all, y_test_all, val_ratio, randomState=random_seed)


    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)

    print('X_val shape: ', X_val.shape)
    print('y_val shape: ', y_val.shape)

    print('X_test shape: ', X_test.shape)
    print('y_test shape: ', y_test.shape)


    print("\n类内样本数量分布:")
    def print_class_distribution(name, labels):
        labels = labels.astype(np.int64)  # 强制转换为整数
        bincount = np.bincount(labels)
        for i, count in enumerate(bincount):
            print(f"{name} 类别 {i}: {count} 个样本")
        print()

    print_class_distribution("训练集", y_train)
    print_class_distribution("验证集", y_val)
    print_class_distribution("测试集", y_test)

    class_weights = compute_class_weights(y_train)
    print("类别权重:", class_weights)



    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    X_train = X_train.reshape(-1, patch_size, patch_size, pca_components, 1)
    X_val = X_val.reshape(-1, patch_size, patch_size, pca_components, 1)
    X_test = X_test.reshape(-1, patch_size, patch_size, pca_components, 1)


    X = X.transpose(0, 4, 3, 1, 2)
    X_train = X_train.transpose(0, 4, 3, 1, 2)
    X_val = X_val.transpose(0, 4, 3, 1, 2)
    X_test = X_test.transpose(0, 4, 3, 1, 2)


    print('after transpose: X shape: ', X.shape)
    print('after transpose: Xtrain shape: ', X_train.shape)
    print('after transpose: X_val shape: ', X_val.shape)
    print('after transpose: Xtest  shape: ', X_test.shape)


    Xset = TestDS(X, y_all)


    trainset = TrainDS(X_train, y_train)

    valset = TestDS(X_val, y_val)


    testset = TestDS(X_test, y_test)


    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator = generator
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


    all_data_loader = torch.utils.data.DataLoader(
        dataset=Xset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


    return train_loader,val_loader, test_loader, all_data_loader, y,class_weights     #只有y没有删除标签0




class SqueezeChannel:
    def __call__(self, x):
        return x.squeeze(0)

class RandomRot90:
    def __call__(self, x):
        if random.random() < 0.5:
            k = random.randint(0, 3)
            return torch.rot90(x, k, dims=[-2, -1])
        return x

class AddGaussianNoise:
    def __call__(self, x):
        if random.random() < 0.5:
            noise = torch.randn_like(x) * 0.01
            return x + noise
        return x

class UnsqueezeChannel:
    def __call__(self, x):
        return x.unsqueeze(0)


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)
        self.training = True

        self.base_transform = transforms.Compose([
            SqueezeChannel()
        ])

        self.train_transform = transforms.Compose([
            RandomRot90(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            AddGaussianNoise(),
            UnsqueezeChannel()
        ])

    def __getitem__(self, index):
        x, y = self.x_data[index], self.y_data[index]

        if self.training:
            x = self.base_transform(x)
            x = self.train_transform(x)

        return x, y

    def __len__(self):
        return self.len

    def set_training(self, training):
        self.training = training


class TestDS(torch.utils.data.Dataset):


    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]

        self.x_data = torch.FloatTensor(Xtest)

        self.y_data = torch.LongTensor(ytest)


    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]


    def __len__(self):
        return self.len



def train_single_gpu(net_model,train_loader, val_loader, epochs, class_weights,model_path):
    device = torch.device("cuda:0")


    net = net_model.to(device)


    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.Adam(net.parameters(), lr=Lr , weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    MIN_loss = float('inf')
    train_losses = []
    val_losses = []


    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0

        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output= net(data, label)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        train_losses.append(epoch_loss)
        print(f"学习率为：{current_lr}")


        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data, val_label in val_loader:
                val_data, val_label = val_data.to(device), val_label.to(device)
                val_output= net(val_data, val_label)
                val_loss_batch = criterion(val_output, val_label)
                val_loss += val_loss_batch.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        val_losses.append(val_loss)


        if val_loss < MIN_loss:
            MIN_loss = val_loss
            torch.save(net.state_dict(), model_path)
        else:
            print(f"Epoch {epoch+1}: 验证损失未下降，当前最优验证损失: {MIN_loss:.4f}")


    print("模型训练完成，已保存")


def Test(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0

    for item in test_loader:
        data, label = item
        data = data.to(device)
        pred = net(data,label)
        pred_class = np.argmax(pred.detach().cpu().numpy(), axis=1)

        if count == 0:
            y_pred_test = pred_class
            y_test = label
            count = 1
        else:

            y_pred_test = np.concatenate((y_pred_test, pred_class))
            y_test = np.concatenate((y_test, label))

    return y_pred_test, y_test


def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)

    list_raw_sum = np.sum(confusion_matrix, axis=1)

    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))

    average_acc = np.mean(each_acc)
    return each_acc, average_acc




def acc_reports(y_test, y_pred_test, name):

    if name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

    if name == 'IP':
        target_names = ['Alfalfa', 'Corn_notill', 'Corn_mintill', 'Corn', 'Pasture', 'Trees', 'Pasture_mowed',
                        'Hay_windrowed', 'Oats','Soybeans_notill','Soybeans_mintill','Soybeans_cleantill','Wheat','Woods','Building_grass','Stone_steel_towers']

    if name == 'BS':
        target_names = ['Water', 'Hippo grass', 'Floodplain Grasses 1', 'Floodplain Grasses 2', 'Reeds1', 'Riparian', 'Firescar 2',
                        'Island interior', 'Acacia woodlands', 'Acacia shrublands', 'Acacia grasslands', 'Short mopane', 'Mixed mopane', 'Exposed soils']




    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)


    oa = accuracy_score(y_test, y_pred_test)


    confusion = confusion_matrix(y_test, y_pred_test)


    each_acc, aa = AA_andEachClassAccuracy(confusion)


    kappa = cohen_kappa_score(y_test, y_pred_test)


    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100



def save_reports(train_time, test_time,y_pred_test,y_test):

    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test, dataset)


    classification = str(classification)


    file_name = "results/classification_report.txt"


    with open(file_name, 'w') as x_file:

        x_file.write('{} Training_Time (s)'.format(train_time))
        x_file.write('\n')


        x_file.write('{} Test_time (s)'.format(test_time))
        x_file.write('\n')


        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')


        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')


        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')


        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')


        x_file.write('{}'.format(classification))
        x_file.write('\n')


        x_file.write('{}'.format(confusion))



def main():

    X, y = loadData(dataset)

    all_results = []
    all_class_accs = []

    for run in range(10):
        print(f"\n========== 第 {run + 1} 次训练 ==========")

        seed = 5555 + run
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        g = torch.Generator()
        g.manual_seed(seed)

        train_loader, val_loader, test_loader, all_data_loader, y_all, class_weights = create_data_loader(
            X, y, patch_size, random_seed=seed ,generator=g
        )

        net_model = model.WTCMC(num_classes=num_classes, inputsize=patch_size, inputdim=pca_components)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = f'params/{dataset}_{timestamp}_seed{seed}_run{run + 1}.pth'

        tic1 = time.perf_counter()
        train_single_gpu(net_model,train_loader, val_loader, Epoech, class_weights,model_path)
        toc1 = time.perf_counter()
        train_time = toc1 - tic1


        print(f"开始第 {run + 1} 次测试")
        device = torch.device("cuda:0")
        net = net_model.to(device)
        net.load_state_dict(torch.load(model_path))
        print("成功加载模型权重")

        tic2 = time.perf_counter()
        y_pred_test, y_test = Test(device, net, test_loader)
        toc2 = time.perf_counter()
        test_time = toc2 - tic2


        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test, dataset)
        print(f"[第 {run + 1} 次] OA: {oa:.2f}%, AA: {aa:.2f}%, Kappa: {kappa:.2f}%")

        save_reports(train_time, test_time, y_pred_test, y_test)


        all_results.append((oa, aa, kappa))
        all_class_accs.append(each_acc)

        get_cls_map.get_cls_map(net, device, all_data_loader, y_all)

    print("\n========== 训练完成：10次结果汇总 ==========")
    oa_sum, aa_sum, kappa_sum = 0.0, 0.0, 0.0
    shu = 0.0
    for i, (oa, aa, kappa) in enumerate(all_results, start=1):
        print(f"第{i}次 => OA: {oa:.2f}%, AA: {aa:.2f}%, Kappa: {kappa:.2f}%")
        oa_sum += oa
        aa_sum += aa
        kappa_sum += kappa
        shu = i

    avg_oa = oa_sum / shu
    avg_aa = aa_sum / shu
    avg_kappa = kappa_sum / shu


    oa_list_biaozhuancha = [r[0] for r in all_results]
    aa_list_biaozhuancha = [r[1] for r in all_results]
    kappa_list_biaozhuancha = [r[2] for r in all_results]

    std_oa = np.std(oa_list_biaozhuancha, ddof=1)
    std_aa = np.std(aa_list_biaozhuancha, ddof=1)
    std_kappa = np.std(kappa_list_biaozhuancha, ddof=1)


    print("\n========== 最终结果（均值 ± 标准差） ==========")
    print(f"OA: {avg_oa:.2f} ± {std_oa:.2f}")
    print(f"AA: {avg_aa:.2f} ± {std_aa:.2f}")
    print(f"Kappa: {avg_kappa:.2f} ± {std_kappa:.2f}")


    all_class_accs = np.array(all_class_accs)
    class_avg_accs = np.mean(all_class_accs, axis=0)
    class_std_accs = np.std(all_class_accs, axis=0, ddof=1)

    if dataset == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    if dataset == 'IP':
        target_names = ['Alfalfa', 'Corn_notill', 'Corn_mintill', 'Corn', 'Pasture', 'Trees', 'Pasture_mowed',
                        'Hay_windrowed', 'Oats','Soybeans_notill','Soybeans_mintill','Soybeans_cleantill','Wheat','Woods','Building_grass','Stone_steel_towers']
    if dataset == 'BS':
        target_names = ['Water', 'Hippo grass', 'Floodplain Grasses 1', 'Floodplain Grasses 2', 'Reeds1', 'Riparian', 'Firescar 2',
                        'Island interior', 'Acacia woodlands', 'Acacia shrublands', 'Acacia grasslands', 'Short mopane', 'Mixed mopane', 'Exposed soils']



    print("\n========== 各类别10次训练的平均准确率 ± 标准差 ==========")
    for i, (avg, std) in enumerate(zip(class_avg_accs, class_std_accs)):
        print(f"{target_names[i]}: {avg:.2f} ± {std:.2f}")



    os.makedirs('results', exist_ok=True)
    with open('results/AVG_result.txt', 'w') as f:

        f.write("========== 所有训练轮次的指标 ==========\n")
        for i, (oa_i, aa_i, kappa_i) in enumerate(all_results, start=1):
            f.write(f"第{i}次 => OA: {oa_i:.2f}%, AA: {aa_i:.2f}%, Kappa: {kappa_i:.2f}%\n")

        f.write("\n========== 10次训练的平均值 ==========\n")
        f.write(f"平均 OA: {avg_oa:.2f}%\n")
        f.write(f"平均 AA: {avg_aa:.2f}%\n")
        f.write(f"平均 Kappa: {avg_kappa:.2f}%\n")

        f.write("\n========== 10次训练的标准差 ==========\n")
        f.write(f"标准差 OA: {std_oa:.2f}\n")
        f.write(f"标准差 AA: {std_aa:.2f}\n")
        f.write(f"标准差 Kappa: {std_kappa:.2f}\n")

        f.write("\n========== 最终结果（均值 ± 标准差） ==========\n")
        f.write(f"OA: {avg_oa:.2f} ± {std_oa:.2f}\n")
        f.write(f"AA: {avg_aa:.2f} ± {std_aa:.2f}\n")
        f.write(f"Kappa: {avg_kappa:.2f} ± {std_kappa:.2f}\n")


        f.write("\n========== 各类别10次训练的平均准确率 ± 标准差 ==========\n")
        for i, (avg, std) in enumerate(zip(class_avg_accs, class_std_accs)):
            f.write(f"{target_names[i]}: {avg:.2f} ± {std:.2f}\n")
        # 保存每个类别的详细数据
        f.write("\n========== 各类别10次训练的原始准确率数据 ==========\n")
        for i, name in enumerate(target_names):
            f.write(f"{name}: {', '.join([f'{acc:.2f}' for acc in all_class_accs[:, i]])}\n")


# 主程序入口
if __name__ == '__main__':
    main()
