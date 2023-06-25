import random
import math
import time
from model import Encoder, Attention, Decoder, Seq2Seq
from prepare_data import Data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def batch_split(x, y, i, batch_size, total_sample_size):
    bond = i+batch_size+1
    extend_constrain = bond > total_sample_size
    src = x[:, i:] if extend_constrain else x[:, i:i+batch_size]
    trg = y[:, i:] if extend_constrain else y[:, i:i+batch_size]

    return src, trg


def train(model, input, gt, optimizer, criterion, clip, batch_size):
    
    model.train()
    epoch_loss = 0
    total_sample_size = len(input[0])
    batch_num = total_sample_size/batch_size
    
    for i in range(0, total_sample_size, batch_size):
            src, trg = batch_split(input, gt, i, batch_size, total_sample_size)

            optimizer.zero_grad()

            output = model(src, trg)

            ret = output[1:, 0, :].squeeze()
            for batch in range(1,output.shape[1]):
                b = output[1:,batch,:].squeeze()
                ret = torch.cat((ret, b), 0)
            output = ret

            output  = torch.softmax(output, dim=1)
            # print(output)

            trg = torch.transpose(trg[1:], 1, 0)
            trg  = torch.flatten(trg)
            # print(trg)

            loss = criterion(output, trg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()

    return epoch_loss / batch_num

def evaluate(model, input, gt, criterion, batch_size):
    
    model.eval()
    epoch_loss = 0
    total_sample_size = len(input[0])
    batch_num = total_sample_size/batch_size

    with torch.no_grad():

        for i in range(0, total_sample_size, batch_size):
            src, trg = batch_split(input, gt, i, batch_size, total_sample_size)
            output = model(src, trg)
            ret = output[1:, 0, :].squeeze()

            for batch in range(1,output.shape[1]):
                b = output[1:,batch,:].squeeze()
                ret = torch.cat((ret, b), 0)
            output = ret
            output  = torch.softmax(output, dim=1)
            # print(output)

            trg = torch.transpose(trg[1:], 1, 0)
            trg  = torch.flatten(trg)
            # print(trg)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / batch_num

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


if __name__ == '__main__':
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    sequence = ["sos M G A E E E D T A I L Y P F T I S G N D R N G N F T I N F K G T P N S T N N G C I G Y S Y N G D W E K I E W E G S C D G N G N L V V E V P M S K I P A G V T S G E I Q I W W H S G D L K M T D Y K A L E H H H H H H eos"
            ,"sos M N K Y L F E L P Y E R S E P G W T I R S Y F D L M Y N E N R F L D A V E N I V N K E S Y I L D G I Y C N F P D M N S Y D E S E H F E G V E F A V G Y P P D E D D I V I V S E E T C F E Y V R L A C E K Y L Q L H P E D T E K V N K L L S K I P S A G H H H H H H eos"
            , "sos M N K Y L F E L P Y E R S E P G W T I R S Y F D L M Y N E N R F L D A V E N I V N K E S Y I L D G I Y C N F P D M N S Y D E S E H F E G V E F A V G Y P P D E D D I V I V S E E T C F E Y V R L A C E K Y L Q L H P E D T E K V N K L L S K I P S A G H H H H H H eos"
            , "sos M G A E E E D T A I L Y P F T I S G N D R N G N F T I N F K G T P N S T N N G C I G Y S Y N G D W E K I E W E G S C D G N G N L V V E V P M S K I P A G V T S G E I Q I W W H S G D L K M T D Y K A L E H H H H H H I eos"
            , "sos M N K Y L F E L P Y E R S E P G W T I R S Y F D L M Y N E N R F L D A V E N I V N K E S Y I L D G I Y C N F P D M N S Y D E S E H F E G V E F A V G Y P P D E D D I V I V S E E T C F E Y V R L A C E K Y L Q L H P E D T E K V N K L L S K I P S A G H H H H H H G eos"]
    function = ['sos The enzyme has a broad substrate specificity, acting on several cyclohexanols and cyclohexenols, and is not identical to (+)-neomenthol dehydrogenase (EC 1.1.1.208). eos'
            ,'sos This enzyme, which is not identical with EC 1.1.1.207 (-)-menthol dehydrogenase, functions by acting on various cyclohexanols and cyclohexenols. eos'
            , 'sos This enzyme, which is not identical with EC 1.1.1.207 (-)-menthol dehydrogenase, functions by acting on various cyclohexanols and cyclohexenols. eos'
            , 'sos This enzyme is found in various organisms and acts on sugar alcohols such as L-iditol, D-glucitol, D-xylitol, and D-galactitol. It shows different substrate specificity depending on the organism or tissue. It is specific to NAD+ and cannot use NADP+. eos'
            , 'sos The enzyme converts xylitol to L-xylulose and L-glucitol to L-fructose. eos']

    sequence_eva = ["sos M N K Y L F E L P Y E R S E P G W T I R S Y F D L M Y N E N R F L D A V E N I V N K E S Y I L D G I Y C N F P D M N S Y D E S E H F E G V E F A V G Y P P D E D D I V I V S E E T C F E Y V R L A C E K Y L Q L H P E D T E K V N K L L S Y I P S A G H H H H H H eos"]
    function_eva = ['sos The enzyme from Agrobacterium fabrum C58 converts alditols with L-threo-configuration adjacent to a primary alcohol group into corresponding sugars, and is involved in D-altritol and galactitol degradation pathways. eos']
    Train_data = Data(sequence, function,sequence_eva, function_eva)
    ## model build 

    INPUT_DIM = Train_data.src_len
    OUTPUT_DIM = Train_data.trg_len
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = 0)

    N_EPOCHS = 2
    CLIP = 1
    batch_size = 2

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, Train_data.src, Train_data.trg, optimizer, criterion, CLIP, batch_size)
        # valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), 'test-model.pt')
        torch.save(model.state_dict(), 'test-model.pt')
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    model.load_state_dict(torch.load('test-model.pt'))

    test_loss = evaluate(model, Train_data.src_eva, Train_data.trg_eva, criterion, batch_size)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')