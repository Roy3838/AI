class TriangleDataset(Dataset):
    def __init__(self, labels, dataset_path, transform=None):
        self.labels = labels
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.dataset_path, self.labels[idx]['file_name'])
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        vertices = np.array(self.labels[idx]['vertices'])
        vertices = vertices.reshape(-1)  # Flatten the vertices array

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'vertices': torch.from_numpy(vertices).float()}
        return sample

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])

dataset = TriangleDataset(labels, dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
