import torch
from torch.utils.data import Dataset, DataLoader

def deserialize_sparse_tensor(d):
    return torch.sparse_coo_tensor(d["indices"], d["values"], d["size"]).coalesce()


class ScattererDataset(Dataset):
    """Dataset for the saved GT sparse tensors and CSI"""
    def __init__(self, data_path, device):
        self.device = device
        self.data = torch.load(data_path, map_location='cpu')

        # deserialize sparse tensors
        self.GT_list = []
        for scale_idx in range(3):  # 3 scales: 0.25,0.5,1
            tensors = [deserialize_sparse_tensor(t) for t in self.data["GT"][scale_idx]]
            self.GT_list.append(tensors)

        # CSI is already dense
        self.CSI_list = [torch.tensor(csi, dtype=torch.float32) for csi in self.data["CSI"]]

        # number of samples
        self.length = len(self.CSI_list)

    def __getitem__(self, idx):
        # return GT tensors (sparse) and CSI tensor (dense)
        GT0 = self.GT_list[0][idx].to(self.device)
        GT1 = self.GT_list[1][idx].to(self.device)
        GT2 = self.GT_list[2][idx].to(self.device)
        CSI = self.CSI_list[idx].to(self.device)

        return {"GT": [GT0, GT1, GT2], "CSI": CSI}

    def __len__(self):
        return self.length


def scatterer_collate_fn(batch):
    """
    Collate function for DataLoader. 
    Since GT are sparse, we keep them as lists.
    """
    GT_batch = [[], [], []]  # 3 scales
    CSI_batch = []

    for b in batch:
        for i in range(3):
            GT_batch[i].append(b["GT"][i])
        CSI_batch.append(b["CSI"])

    return {"GT": GT_batch, "CSI": torch.stack(CSI_batch, dim=0)}


def create_scatterer_dataloader(path, batch_size, num_workers, device, val_only=False):
    filenames = ['save_res_train.pt', 'save_res_val.pt', 'save_res_test.pt'] if not val_only else ['save_res_test.pt']
    dataloaders = []

    for name in filenames:
        dataset = ScattererDataset(data_path=f"{path}/{name}", device=device)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True if 'train' in name else False,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=scatterer_collate_fn)
        dataloaders.append(loader)
    return dataloaders

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = create_scatterer_dataloader(
        "/home/jingpeng/graduation_project/3DCSIYOLO/data",
        batch_size=4,
        num_workers=4,
        device=device,
        val_only=True
    )

    for batch in test_loader:
        # batch["GT"] -> list of 3 lists (sparse tensors)
        # batch["CSI"] -> dense tensor [B, 64, 64, 64]
        #print(len(batch["GT"]), batch["CSI"].shape)
        print(len(batch))
        for b in batch:
            print(b["GT"][0][0],b["GT"][1][0],b["GT"][2][0])
            print(torch.tensor(b["CSI"]).shape)
            break
        break

