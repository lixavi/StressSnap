import torch
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResNetWithAttention(nn.Module):
    def __init__(self):
        super(ResNetWithAttention, self).__init__()
        
        self.resnet = models.resnet152(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(2048, 7)
        
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.features(x)
        attention = self.attention(features)
        features = features * attention
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output


class ResNetWithMAttention(nn.Module):
    def __init__(self):
        super(ResNetWithMAttention, self).__init__()
        
        self.resnet = models.resnet152(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(2048, 7)
        
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.features(x)
        attention = self.attention(features)
        features = features * attention
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output

class ResNet152Mod(nn.Module):
    def __init__(self):
        super(ResNet152Mod, self).__init__()
        self.model = models.resnet152(pretrained=True).to(device)
        self.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 7)).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = True  
        
    def forward(self, x):
        output = self.model(x)
        output = output.view(output.size(0), -1) # flatten the output tensor
        output = self.fc(output)
        return output

class ResNet101Mod(nn.Module):
    def __init__(self):
        super(ResNet101Mod, self).__init__()
        self.model = models.resnet101(pretrained=True).to(device)
        self.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 7)).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = True  
        
    def forward(self, x):
        output = self.model(x)
        output = output.view(output.size(0), -1) # flatten the output tensor
        output = self.fc(output)
        return output


class ResNet50Mod(nn.Module):
    def __init__(self):
        super(ResNet50Mod, self).__init__()
        self.model = models.resnet50(pretrained=True).to(device)
        self.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 7)).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = True  
        
    def forward(self, x):
        output = self.model(x)
        output = output.view(output.size(0), -1) # flatten the output tensor
        output = self.fc(output)
        return output

        
class VGG16Mod(nn.Module):
    def __init__(self):
        super(VGG16Mod, self).__init__()
        self.model = models.vgg16(pretrained=True)
        
        for param in self.model.parameters():
            param.requires_grad = True  
        
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, 7)
        
    def forward(self, x):
        output = self.model(x)
        return output


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch, model):
        images, labels = batch 
        out = model(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs, model):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))