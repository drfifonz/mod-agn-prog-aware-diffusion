"""Image-only transforms; patient membership is fixed before loading pixels."""
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class RecordDataset(Dataset):
    def __init__(self,records,image_size,modality,augment=False,standardize=True):
        self.records=list(records)
        operations=[]
        if modality=='oct':
            # Generated OCT images are already in model coordinates.
            self.real_crop=T.CenterCrop((256,512))
        else:self.real_crop=None
        operations += [T.Resize(tuple(image_size))]
        if augment:operations.append(T.RandomHorizontalFlip(.5))
        operations += [T.Grayscale(num_output_channels=1),T.ToTensor()]
        if standardize:operations.append(T.Normalize((.5,),(.5,)))
        self.transform=T.Compose(operations)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        record=self.records[index]
        with Image.open(record.path) as image:image=image.convert('L')
        if self.real_crop is not None and record.source=='real':image=self.real_crop(image)
        return self.transform(image),self.class_to_index[record.label] if hasattr(self,'class_to_index') else 0
