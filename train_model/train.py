import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        self.layer1 = nn.Conv2d(1, 32, 3, 1)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.drop_a = nn.Dropout(0.25)
        self.drop_b = nn.Dropout(0.5)
        self.dense1 = nn.Linear(9216, 128)
        self.dense2 = nn.Linear(128, 10)

    def forward(self, input_img):
        input_img = self.layer1(input_img)
        input_img = F.relu(input_img)
        input_img = self.layer2(input_img)
        input_img = F.relu(input_img)
        input_img = F.max_pool2d(input_img, 2)
        input_img = self.drop_a(input_img)
        input_img = torch.flatten(input_img, 1)
        input_img = self.dense1(input_img)
        input_img = F.relu(input_img)
        input_img = self.drop_b(input_img)
        input_img = self.dense2(input_img)
        return F.log_softmax(input_img, dim=1)


# def run_training(params, net, dev, loader, opt, current_epoch):
#     print("Training started...\n")
#     net.train()
#     for batch_idx, (x, y) in enumerate(loader):
#         x, y = x.to(dev), y.to(dev)
#         opt.zero_grad()
#         preds = net(x)
#         loss_val = F.nll_loss(preds, y)
#         loss_val.backward()
#         opt.step()
#         if batch_idx % params.log_interval == 0:
#             print(f'Train Epoch: {current_epoch} [{batch_idx * len(x)}/{len(loader.dataset)} '
#                   f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss_val.item():.6f}')
#             if params.dry_run:
#                 break

def train(model, device, dataloader, optimizer, epoch, params):
    model.train()
    print("\n[INFO] Starting Training Epoch...\n")
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if batch_idx % params.log_interval == 0:
            print(f"[Train] Epoch {epoch} | Batch {batch_idx * len(inputs)}/{len(dataloader.dataset)} "
                  f"({100. * batch_idx / len(dataloader):.0f}%) | Loss: {loss.item():.6f}")
            
            if params.dry_run:
                print("[INFO] Dry run enabled — stopping after first batch.\n")
                break

# testing 
# def run_evaluation(net, dev, eval_loader):
#     net.eval()
#     total_loss = 0
#     correct_preds = 0
#     with torch.no_grad():
#         for x, y in eval_loader:
#             x, y = x.to(dev), y.to(dev)
#             outputs = net(x)
#             total_loss += F.nll_loss(outputs, y, reduction='sum').item()
#             guess = outputs.argmax(dim=1, keepdim=True)
#             correct_preds += guess.eq(y.view_as(guess)).sum().item()
#     total_loss /= len(eval_loader.dataset)
#     print(f'\nTest set: Average loss: {total_loss:.4f}, '
#           f'Accuracy: {correct_preds}/{len(eval_loader.dataset)} '
#           f'({100. * correct_preds / len(eval_loader.dataset):.0f}%)\n')

def evaluate(model, device, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += F.nll_loss(outputs, targets, reduction='sum').item()
            predictions = outputs.argmax(dim=1, keepdim=True)
            correct += predictions.eq(targets.view_as(predictions)).sum().item()
    
    average_loss = total_loss / len(dataloader.dataset)
    accuracy = 100. * correct / len(dataloader.dataset)
    print(f"\n[Eval] Average Loss: {average_loss:.4f} | Accuracy: {correct}/{len(dataloader.dataset)} ({accuracy:.2f}%)\n")
    
    return average_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Digit Classifier - PyTorch MNIST')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-accel', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true')
    args = parser.parse_args()

    use_cuda = not args.no_accel and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if use_cuda else "cpu")

    train_config = {'batch_size': args.batch_size}
    eval_config = {'batch_size': args.test_batch_size}
    if use_cuda:
        loader_opts = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_config.update(loader_opts)
        eval_config.update(loader_opts)

    preprocess_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=preprocess_mnist)
    train_data_loader = torch.utils.data.DataLoader(train_data, **train_config)

    digit_classifier = DigitNet().to(dev)
    optimizer = optim.Adadelta(digit_classifier.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for ep in range(1, args.epochs + 1):
        run_training(args, digit_classifier, dev, train_data_loader, optimizer, ep)
        run_evaluation(digit_classifier, dev, test_loader)  
        lr_scheduler.step()

    if args.save_model:
        print("model saved at /mnt/gd2574_model.pt")
        torch.save(digit_classifier.state_dict(), "/mnt/gd2574_model.pt")

if __name__ == '__main__':
    main()
