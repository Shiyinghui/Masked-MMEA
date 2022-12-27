# @Time   : 2022/9/26 13:20
# @Author : syh
from utils import *
from model import *
import gc
import config
import torch.optim as optim
import os
import time


def load_data():
    lang_list = [1, 2]
    ent2id_dict, ills, triples, r_hs, r_ts, ids = read_raw_data(config.data_dir, lang_list)
    e1 = os.path.join(config.data_dir, 'ent_ids_1')
    e2 = os.path.join(config.data_dir, 'ent_ids_2')
    left_ents = get_ids(e1)
    right_ents = get_ids(e2)
    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(r_hs)
    adj = get_adjr(ENT_NUM, triples, norm=True)
    img_features, _ = load_img(ENT_NUM, config.img_feature_path)
    return ent2id_dict, ENT_NUM, REL_NUM, left_ents, right_ents, ills, adj, img_features


def set_seed():
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def model_test(model, epoch, input_idx, adj_matrix,
               test_left, test_right, img_feats, ent_mask, mat_mask):
    with torch.no_grad():
        model.eval()

        struct_emb, img_emb = model(input_idx, adj_matrix, img_feats, ent_mask)
        #struct_emb, img_emb = model(input_idx, adj_matrix, img_feats)
        struct_emb = F.normalize(struct_emb)
        img_emb = F.normalize(img_emb)

        top_k = [1, 10]
        acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
        acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
        test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.

        # struct_sim = struct_emb[test_left].mm(struct_emb[test_right].t())
        # final_sim = csls_sim(struct_sim, config.csls_k)
        # preds = torch.argmax(struct_sim, dim=1).cpu().numpy().tolist()
        # _, preds = torch.sort(struct_sim, 1, descending=True)

        weight = [0.5, 0.5]
        struct_sim = struct_emb[test_left].mm(struct_emb[test_right].t())
        vis_sim = img_emb[test_left].mm(img_emb[test_right].t())
        #
        struct_sim[torch.where(mat_mask == 1)] *= weight[0]
        vis_sim *= weight[1]

        # vis_sim[torch.where(mat_mask == 0)] = 0
        # vis_sim[torch.where(mat_mask == 1)] *= weight[1]

        combined_sim = struct_sim + vis_sim
        final_sim = csls_sim(combined_sim, config.csls_k)

        if epoch + 1 == config.epochs:
            to_write = []
            test_left_np = test_left.cpu().numpy()
            test_right_np = test_right.cpu().numpy()
            to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3"])

        for idx in range(test_left.shape[0]):
            values, indices = torch.sort(final_sim[idx, :], descending=True)
            rank = (indices == idx).nonzero().squeeze().item()
            mean_l2r += (rank + 1)
            mrr_l2r += 1.0 / (rank + 1)
            for i in range(len(top_k)):
                if rank < top_k[i]:
                    acc_l2r[i] += 1
            if epoch + 1 == config.epochs:
                indices = indices.cpu().numpy()
                to_write.append([idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]],
                                 test_right_np[indices[1]], test_right_np[indices[2]]])

        if epoch + 1 == config.epochs:
            import csv
            with open(config.res_path, "w") as f:
                wr = csv.writer(f, dialect='excel')
                wr.writerows(to_write)

        mean_l2r /= test_left.size(0)
        mrr_l2r /= test_left.size(0)
        for i in range(len(top_k)):
            acc_l2r[i] = round(acc_l2r[i] / test_left.size(0), 4)
            acc_r2l[i] = round(acc_r2l[i] / test_right.size(0), 4)

        print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}".format(top_k, acc_l2r, mean_l2r, mrr_l2r))
        del final_sim, struct_emb, img_emb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_test_mask(left_indices, right_indices, mask):
    left = mask[left_indices].repeat(len(left_indices), 1).t()
    right = mask[right_indices].repeat(len(right_indices), 1)
    # left = mask[left_indices]
    # print(len(left[left == 1]))
    # right = mask[right_indices]
    # print(len(right[right == 1]))
    mat_mask = (left * right).int()
    return mat_mask


def train(model, optimizer, train_indices, test_indices,
          input_idx, adj_matrix, device, img_feats, mask):

    test_left = torch.LongTensor(test_indices[:, 0].squeeze()).to(device)
    test_right = torch.LongTensor(test_indices[:, 1].squeeze()).to(device)

    ent_mask = mask
    mat_mask = get_test_mask(test_left, test_right, mask)

    loss_func_gcn = NCA_loss(alpha=5, beta=10, ep=0.0)
    loss_func_img = masked_nca_loss(alpha=15, beta=10, ep=0.0)
    #loss_func_img = NCA_loss(alpha=15, beta=10, ep=0.0)

    print("[start training...]")
    for epoch in range(config.epochs):
        start_time = time.time()
        np.random.shuffle(train_indices)
        model.train()
        optimizer.zero_grad()
        struct_emb, img_emb = model(input_idx, adj_matrix, img_feats, mask)
        #struct_emb, img_emb = model(input_idx, adj_matrix, img_feats)
        # mm_emb = torch.cat([F.normalize(struct_emb).detach(),
        #                     F.normalize(img_emb).detach()],
        #                    dim=1)

        loss_epoch, loss_batch = 0, 0
        for si in np.arange(0, train_indices.shape[0], config.batch_size):
            loss_gcn = loss_func_gcn(struct_emb, train_indices[si:si+config.batch_size], [], device=device)
            loss_img = loss_func_img(img_emb, train_indices[si:si+config.batch_size], [], mask, device=device)
            #loss_img = loss_func_img(img_emb, train_indices[si:si + config.batch_size], [], device=device)
            loss_batch = loss_gcn + loss_img
            # loss_mm = loss_func_mm(mm_emb, train_indices[si:si+config.batch_size], [], device=device)
            # loss_batch = loss_gcn + loss_img + loss_mm
            loss_batch.backward(retain_graph=True)
            loss_epoch += loss_batch.item()

        optimizer.step()
        print("[epoch {:d}] loss_all: {:f}, time: {:.4f} s".format(epoch, loss_epoch, time.time()-start_time))
        del struct_emb, img_emb

        if (epoch + 1) % config.check_point == 0:
            print("\n[epoch {:d}] checkpoint!".format(epoch))
            model_test(model, epoch, input_idx, adj_matrix,
                       test_left, test_right, img_feats, ent_mask, mat_mask)


def get_mask(ent2id_dict, ent_num, file_path=None):
    if file_path is None:
        mask = torch.ones(ent_num, dtype=torch.float32)
        print(len(mask[mask == 1]))
    else:
        mask = torch.zeros(ent_num, dtype=torch.float32)
        with open(file_path, 'rb') as f:
            info: dict = pickle.load(f)
        for k, v in info.items():
            mask[ent2id_dict[k]] = 1 if v > 0 else 0
        print(len(mask[mask==1]))
    return mask


def main():
    set_seed()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ent2id_dict, ENT_NUM, REL_NUM, left_ents, right_ents, ills, adj, img_features = load_data()

    np.random.shuffle(ills)
    train_ill = np.array(ills[:int(len(ills) // 1 * config.train_rate)], dtype=np.int32)
    test_ill = np.array(ills[int(len(ills) // 1 * config.train_rate):], dtype=np.int32)

    # move data to gpu
    input_ent_idx = torch.LongTensor(np.arange(ENT_NUM)).to(device)
    adj = adj.to(device)
    img_features = F.normalize(torch.Tensor(img_features).to(device))

    # set model
    mask = get_mask(ent2id_dict, ENT_NUM, config.mask_path).to(device)
    #mask = get_mask(ent2id_dict, ENT_NUM).to(device)
    gcn_units = config.gcn_hidden_units

    img_feature_dim = img_features.shape[1]
    img_emb_dim = config.img_emb_dim
    ea_model = MASK_MMEA(ENT_NUM, gcn_units, config.dropout,
                         img_feature_dim=img_feature_dim,
                         img_emb_dim=img_emb_dim)
    ea_model = ea_model.to(device)
    optimizer = optim.AdamW(ea_model.parameters(), lr=config.lr)

    # start training
    train(model=ea_model,
          optimizer=optimizer,
          train_indices=train_ill,
          test_indices=test_ill,
          input_idx=input_ent_idx,
          adj_matrix=adj,
          device=device,
          img_feats=img_features,
          mask=mask)


if __name__ == "__main__":
    main()


