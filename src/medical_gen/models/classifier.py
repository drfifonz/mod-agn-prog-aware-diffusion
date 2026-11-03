"""Single-device ResNet34 evaluation with global, epoch-level macro-F1."""
import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.models import resnet34,ResNet34_Weights
from ..metrics import classification_metrics

class ResNetClassifier(pl.LightningModule):
    def __init__(self,classes,pretrained=True,learning_rate=1e-4):
        super().__init__();self.save_hyperparameters()
        self.classes=list(classes);self.learning_rate=learning_rate
        self.model=resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        old=self.model.conv1
        self.model.conv1=nn.Conv2d(1,old.out_channels,kernel_size=old.kernel_size,stride=old.stride,padding=old.padding,bias=False)
        if pretrained:
            with torch.no_grad():self.model.conv1.weight.copy_(old.weight.mean(dim=1,keepdim=True))
        self.model.fc=nn.Linear(self.model.fc.in_features,len(classes))
        self.loss=nn.CrossEntropyLoss()
        self.val_true=[];self.val_pred=[];self.test_true=[];self.test_pred=[]

    def forward(self,images):return self.model(images)

    def training_step(self,batch,batch_idx):
        x,y=batch;loss=self.loss(self(x),y)
        self.log('train_loss',loss,on_step=False,on_epoch=True,batch_size=len(y))
        return loss

    def on_validation_epoch_start(self):self.val_true=[];self.val_pred=[]

    def validation_step(self,batch,batch_idx):
        x,y=batch;logits=self(x);loss=self.loss(logits,y)
        self.log('val_loss',loss,on_step=False,on_epoch=True,batch_size=len(y))
        self.val_true.extend(y.cpu().tolist());self.val_pred.extend(logits.argmax(1).cpu().tolist())

    def on_validation_epoch_end(self):
        if self.val_true:self.log('val_macro_f1',classification_metrics(self.val_true,self.val_pred,self.classes)['macro_f1'])

    def on_test_epoch_start(self):self.test_true=[];self.test_pred=[]

    def test_step(self,batch,batch_idx):
        x,y=batch;logits=self(x)
        self.test_true.extend(y.cpu().tolist());self.test_pred.extend(logits.argmax(1).cpu().tolist())

    def on_test_epoch_end(self):
        self.test_metrics=classification_metrics(self.test_true,self.test_pred,self.classes)
        self.log('test_macro_f1',self.test_metrics['macro_f1']);self.log('test_accuracy',self.test_metrics['accuracy'])

    def configure_optimizers(self):return torch.optim.Adam(self.parameters(),lr=self.learning_rate)
