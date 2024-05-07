import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from train_model import Net
import time  # импорт модуля для работы с временем

# Загрузка обученной модели
net = Net()
net.load_state_dict(torch.load('./mnist_net.pth'))

# Подготовка тестового датасета
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = tuple(str(i) for i in range(10))


# Визуализация нескольких изображений и их предсказаний
def imshow(img):
    img = img / 2 + 0.5  # денормализация
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(testloader)
for images, labels in dataiter:
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

    time.sleep(1)  # добавление задержки в одну секунду
