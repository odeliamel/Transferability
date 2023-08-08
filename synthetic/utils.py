import torch


def calc_loss(args, batch, output):
    if args.train_loss == 'exp':
        loss = torch.sum(torch.exp(-1 * torch.mul(batch[1].float(), output.T)))
    elif args.train_loss == 'log':
        loss = torch.sum(torch.log(1 + torch.exp(-1 * torch.mul(batch[1].float(), output.T))))
    elif args.train_loss == 'square':
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(batch[1].float(), output)

    return loss


def calc_adv_loss(args, batch, output):
    if args.adv_loss == 'margin':
        # y = batch[1].squeeze(1)
        # print(batch[1].shape, output.shape)
        # print(torch.exp(-1 * torch.mul(y.float(), output.T)).shape)
        # loss = torch.sum(torch.exp(-1 * torch.mul(y.float(), output)))
        # loss = torch.sum(torch.log(1 + torch.exp(-1 * torch.mul(batch[1].float(), output.T))))
        loss = torch.sum(torch.mul(((-1)*(batch[1])), output.T))
        # print(batch[1].shape, output.shape)
        # print(batch[1], output, torch.mul(((-1)*(batch[1])), output.T))

    return loss


def evaluate(args, model, dataloader, epoch):
    total_loss = 0
    total_len = 0
    correct_label_count = 0
    model.eval()
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.cuda()
        target = target.cuda()
        total_len += target.shape[0]
        output = model(data.float())
        loss = calc_loss(args, (data, target), output)
        total_loss += loss.detach().item()
        # print('Loss: ', loss)
        for i in range(len(output)):
            if (output[i] * target[i] > 0).item():
                correct_label_count += 1
        # print('Correct label: ', correct_label_count, '/', len(output))
    print('epoch: ', epoch, ' Loss: ', total_loss / total_len, ' Correct label: ', correct_label_count, '/', total_len)
    return correct_label_count/total_len



def get_batch_size(args):
    if args.train_batch_size == 0:
        return args.data_amount
    else:
        return args.train_batch_size