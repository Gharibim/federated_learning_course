import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

def mnistIID(dataset, num_users):
    images = int(len(dataset)/num_users)
    users_dict, indeces = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(i)
        users_dict[i] = set(np.random.choice(indeces, images, replace=False))
        indeces = list(set(indeces) - users_dict[i])
    return users_dict

def mnistNonIID(dataset, num_users, test=False):
    classes, images = 200, 300
    if test:
        classes, images = 20, 500
    classes_indx = [i for i in range(classes)]
    users_dict = {i: np.array([]) for i in range(num_users)}
    indeces = np.arange(classes*images)
    unsorted_labels = dataset.train_labels.numpy()

    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :]

    for i in range(num_users):
        np.random.seed(i)
        temp = set(np.random.choice(classes_indx, 2, replace=False))
        classes_indx = list(set(classes_indx) - temp)
        for t in temp:
            users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)
    return users_dict

def mnistNonIIDUnequal(dataset, num_users, test=False):
    classes, images = 1200, 50
    if test:
        classes, images = 200, 50
    classes_indx = [i for i in range(classes)]
    users_dict = {i: np.array([]) for i in range(num_users)}
    indeces = np.arange(classes*images)
    unsorted_labels = dataset.train_labels.numpy()

    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :]

    min_cls_per_client = 1
    max_cls_per_client = 30

    random_selected_classes = np.random.randint(min_cls_per_client, max_cls_per_client+1, size=num_users)
    random_selected_classes = np.around(random_selected_classes / sum(random_selected_classes) * classes)
    random_selected_classes = random_selected_classes.astype(int)

    if sum(random_selected_classes) > classes:
        for i in range(num_users):
            np.random.seed(i)
            temp = set(np.random.choice(classes_indx, 1, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)

        random_selected_classes = random_selected_classes-1

        for i in range(num_users):
            if len(classes_indx) == 0:
                continue
            class_size = random_selected_classes[i]
            if class_size > len(classes_indx):
                class_size = len(classes_indx)
            np.random.seed(i)
            temp = set(np.random.choice(classes_indx, class_size, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)
    else:

        for i in range(num_users):
            class_size = random_selected_classes[i]
            np.random.seed(i)
            temp = set(np.random.choice(classes_indx, class_size, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)

        if len(classes_indx) > 0:
            class_size = len(classes_indx)
            j = min(users_dict, key=lambda x: len(users_dict.get(x)))
            temp = set(np.random.choice(classes_indx, class_size, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[j] = np.concatenate((users_dict[j], indeces[t*images:(t+1)*images]), axis=0)

    return users_dict

def load_dataset(num_users, iidtype):
    tranform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./', train=True, download=True, transform=tranform)
    test_dataset = datasets.MNIST('./', train=False, download=True, transform=tranform)
    train_group, test_group = None, None
    if iidtype == 'iid':
        train_group = mnistIID(train_dataset, num_users)
        test_group = mnistIID(test_dataset, num_users)
    elif iidtype == 'noniid':
        train_group = mnistNonIID(train_dataset, num_users)
        test_group = mnistNonIID(test_dataset, num_users, True)
    else:
        train_group = mnistNonIIDUnequal(train_dataset, num_users)
        test_group = mnistNonIIDUnequal(test_dataset, num_users, True)
    return train_dataset, test_dataset, train_group, test_group

class FedDataset(Dataset):
    def __init__(self, dataset, indx):
        self.dataset = dataset
        self.indx = [int(i) for i in indx]
        
    def __len__(self):
        return len(self.indx)
    
    def __getitem__(self, item):
        images, label = self.dataset[self.indx[item]]
        return torch.tensor(images).clone().detach(), torch.tensor(label).clone().detach()

def getActualImgs(dataset, indeces, batch_size):
    return DataLoader(FedDataset(dataset, indeces), batch_size=batch_size, shuffle=True)