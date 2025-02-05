

def batch_data_to_device(data, device):
    batch_x, y = data
    y = y.to(device)

    for it in range(0, len(batch_x)):
        if it == len(batch_x) - 1:
            x = batch_x[it]
            x_len = len(x[0])
            for i in range(0, len(x)):
                for j in range(0, x_len):
                    x[i][j] = x[i][j].to(device)
            batch_x[it] = x
        else:
            batch_x[it] = batch_x[it].to(device)
    return [batch_x, y]
