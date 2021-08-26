from torch.nn.modules.loss import MSELoss
import mymodels
import mydataloaders
import pretrained
from torch import optim
import torch

train_loader, test_loader = mydataloaders.create_catdog_loaders()
model = pretrained.load_densenet(2)


optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
criterion = torch.nn.modules.loss.NLLLoss()

# Currently we're on the CPU, but if we get an nVidia CUDA card we can use it
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device) # If we had a CUDA device, this would make our processing way faster
        # write the training loop. call the loss "loss" so that the line below will work
        label_predictions = model.forward(inputs)
        loss = criterion(label_predictions, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            # REMEMBER TO ACTIVATE THE EVAL MODE
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    # Accuracy Calculations
                    top_p, top_class = logps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))                  
                   
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(test_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_loader):.3f}")
            running_loss = 0
            # Return to training
            model.train()




