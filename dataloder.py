import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, csv_file_path, root_dir, transform=None):
        # csvデータの読み出し
        image_dataframe = []
        with open(csv_file_path, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                image_dataframe.append(row)
        self.image_dataframe = image_dataframe
        self.root_dir = root_dir
        # 入力データへの加工
        self.transform = transform
        
    # データのサイズ
    def __len__(self):
        return len(self.image_dataframe)
    
    # データとラベルの取得
    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        label = self.image_dataframe[idx][0]
        img_name = os.path.join(self.root_dir, self.image_dataframe[idx][1]))
        image = Image.open(filename)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label