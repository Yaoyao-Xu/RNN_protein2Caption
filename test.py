import torch

def evaluate(model, src, trg, criterion, batch_size):
    
    model.eval()
    epoch_loss = 0
    total_sample_size = len(src[0])
    batch_num = total_sample_size/batch_size

    with torch.no_grad():

        for i in range(0, total_sample_size):
            bond = i+batch_size+1
            extend_constrain = bond > total_sample_size 
            src = src[:][i:] if extend_constrain else src[:][i:i+batch_size]
            trg = trg[:][i:] if extend_constrain else trg[:][i:i+batch_size]

            output = model(src, trg, 0) #turn off teacher forcing
            output = output[1:].permute(1,0,2)
            ret = output[:][0][:].squeeze()
            for batch in range(1,batch_size):
                b = output[:][batch][:].squeeze()
                ret = torch.cat((ret, b), 0)
            output = ret
            output  = torch.softmax(output, dim=1)
            print('the output of the model is: {}'.format(output))

            trg = torch.transpose(trg[1:], 1, 0)
            trg  = torch.flatten(trg)
            print('the reshaped trg of the is: {}'.format(trg))

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / batch_num



def train(model, src, trg, optimizer, criterion, clip):
    
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()
    output = model(src, trg)
    output = output[1:].permute(1,0,2)

    ret = output[:][0][:].squeeze()
    for batch in range(1,2):
      b = output[:][batch][:].squeeze()
      ret = torch.cat((ret, b), 0)
    output = ret
    output  = torch.softmax(output, dim=1)
    print('the output of the model is: {}'.format(output))

    trg = torch.transpose(trg[1:], 1, 0)
    trg  = torch.flatten(trg)
    print('the reshaped trg of the is: {}'.format(trg))

    loss = criterion(output, trg)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()

    epoch_loss += loss.item()
    return epoch_loss/2

