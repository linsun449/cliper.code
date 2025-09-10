from data.ade.ade_datasets import ADESegDataset
from data.cityscapes.city_datasets import CitySegDataset
from data.coco14.coco_datasets import COCOSegDataset, COCOStuffSegDataset
from data.voc.voc_datasets import VOC12SegmentationDataset


def load_dataset(cfg, model):
    if cfg.dataset_name == "voc21":
        dataset = VOC12SegmentationDataset(img_name_list_path=cfg.img_name_list_path,
                                           label_dir=cfg.label_dir,
                                           voc_root=cfg.voc12_root,
                                           img_transform=model.preprocess)
        bg_text_features = model.classifier(dataset.background, cfg.semantic_templates)
        fg_text_features = model.classifier(dataset.categories, cfg.semantic_templates)

    elif cfg.dataset_name == "context":
        dataset = VOC12SegmentationDataset(img_name_list_path=cfg.img_name_list_path,
                                           label_dir=cfg.label_dir,
                                           voc_root=cfg.voc12_root,
                                           img_transform=model.preprocess)
        bg_text_features = model.classifier(dataset.background, cfg.semantic_templates)
        fg_text_features = model.classifier(dataset.categories, cfg.semantic_templates)
    elif cfg.dataset_name == "coco5k":
        dataset = COCOSegDataset(img_name_list_path=cfg.img_name_list_path,
                                 coco_root=cfg.coco_root,
                                 used_dir=cfg.used_dir,
                                 img_transform=model.preprocess)
        bg_text_features = model.classifier(dataset.background, cfg.semantic_templates)
        fg_text_features = model.classifier(dataset.categories, cfg.semantic_templates)
    elif cfg.dataset_name == "coco_stuff164":
        dataset = COCOStuffSegDataset(img_name_list_path=cfg.img_name_list_path,
                                      coco_root=cfg.coco_root,
                                      used_dir=cfg.used_dir,
                                      img_transform=model.preprocess)
        bg_text_features = model.classifier(dataset.background, cfg.semantic_templates)
        fg_text_features = model.classifier(dataset.categories, cfg.semantic_templates)
    elif cfg.dataset_name == "ade150":
        dataset = ADESegDataset(img_name_list_path=cfg.img_name_list_path,
                                ade_root=cfg.ade_root,
                                used_dir=cfg.used_dir,
                                img_transform=model.preprocess)
        bg_text_features = model.classifier(dataset.background, cfg.semantic_templates)
        fg_text_features = model.classifier(dataset.categories, cfg.semantic_templates)
    elif cfg.dataset_name == "cityscapes":
        dataset = CitySegDataset(img_name_list_path=cfg.img_name_list_path,
                                 city_root=cfg.city_root,
                                 used_dir=cfg.used_dir,
                                 img_transform=model.preprocess)
        bg_text_features = model.classifier(dataset.background, cfg.semantic_templates)
        fg_text_features = model.classifier(dataset.categories, cfg.semantic_templates)
    else:
        raise NotImplementedError("Unknown dataset")
    return dataset, bg_text_features, fg_text_features
