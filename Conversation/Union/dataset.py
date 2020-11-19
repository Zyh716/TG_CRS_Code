from torch.utils.data import Dataset
import torch


class GPT2Dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index, sep_id=102):
        # mask_r: response是1，其他都是0
        input_ids = self.data_list[index].strip()
        input_ids = [int(token_id) for token_id in input_ids.split()]
        mask_r = []
        is_r = True
        flag = False
        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == 102:
                if flag:
                    is_r = False
                if not flag:
                    flag = True
            if is_r:
                mask_r.append(1)
            else:
                mask_r.append(0)

        return input_ids, mask_r[::-1]

    def __len__(self):
        return len(self.data_list)
