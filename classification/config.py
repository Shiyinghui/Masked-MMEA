import os
import pickle


class Config:
    def __init__(self, split='fr_en'):
        assert split in ['fr_en', 'ja_en', 'zh_en']
        ds = split  # 'fr_en', 'ja_en', 'zh_en'
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.ds = split
        self.base_dir = project_dir
        
        data_dir = os.path.join(project_dir, f'data/DBP15K/{ds}/')
        self.kg1_ent2id_path = data_dir + 'ent_ids_1'
        self.kg2_ent2id_path = data_dir + 'ent_ids_2'
        self.ills_path = data_dir + 'ill_ent_ids'
        self.kg1_triple_path = data_dir + 'triples_1'
        self.kg2_triple_path = data_dir + 'triples_2'

        self.ent_type_dir = os.path.join(project_dir, 'data/DBP15K_type/')
        self.ontology_file_path = self.ent_type_dir + 'onto_subClassOf_triples'
        self.disjoint_info = self.ent_type_dir + 'disjoint.txt'
        
        self.vis_encoder = 'resnet152'
        self.img_data_dir = os.path.join(project_dir, 'dbp_images')
        #self.img_data_dir = '/code/syh/dbp_images'
        self.ent2imgfn_path = os.path.join(project_dir, 'img_data/ent2imgfn.pkl')
        self.vis_dir = os.path.join(project_dir, 'img_data', self.vis_encoder, f'{ds}/')
        
        self.model_path = self.vis_dir + f'{ds}_model.pth'
        self.train_img_data_path = self.vis_dir + f'{ds}_train.pkl'
        # ill: inter-language links, ill_img_data used for test
        self.ill_img_data_path = self.vis_dir + f'{ds}_test.pkl'

        self.img_feature_path = self.vis_dir + f'{ds}_feature.pkl'
        self.mask_path = self.vis_dir + f'{ds}_mask.pkl'
        self.pred_path = self.vis_dir + f'{ds}_pred.pkl'
        self.id2type_path = self.vis_dir + f'{ds}_id2type.pkl'

        self.sv_mask_path = self.vis_dir + f'{ds}_mask_1.0.pkl'
        self.spec_mask_path = self.vis_dir + f'{ds}_spec_mask.pkl'

        if os.path.exists(self.ill_img_data_path):
            data = pickle.load(open(self.ill_img_data_path, 'rb'))
            self.finetune_classes = len({item[3] for item in data})


# if __name__ == "__main__":
#     cfg = Config('fr_en')
