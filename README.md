# ContrastiveAA
The code implementation of our WWW'24 SocialNLP paper "Contrastive Disentanglement for Authorship Attribution"

Please leave issues for any questions about the paper or the code.

If you find our code or paper useful, please cite the paper:
```
@inproceedings{10.1145/3589335.3652501,
author = {Hu, Zhiqiang and Nguyen, Thao Thanh and Hu, Yujia and Hung, Chia-Yu and Hee, Ming Shan and Seah, Chun Wei and Lee, Roy Ka-Wei},
title = {Contrastive Disentanglement for Authorship Attribution},
year = {2024},
isbn = {9798400701726},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3589335.3652501},
doi = {10.1145/3589335.3652501},
abstract = {Authorship Attribution (AA) seeks to determine the authorship of texts by examining distinctive writing styles. Although current AA methods have shown promising results, they often underperform in scenarios with significant topic shifts. This limitation arises from their inability to effectively separate topical content from the author's stylistic elements. Furthermore, most studies have focused on individual-level AA, overlooking the potential of regional-level AA to uncover linguistic patterns influenced by cultural and geographical factors. To bridge these gaps, this paper introduces ContrastDistAA, a novel framework that leverages contrastive learning and mutual information maximization to disentangle content and stylistic features in latent representations for AA. Our extensive experiments demonstrate that ContrastDistAA surpasses existing state-of-the-art models in both individual and regional-level AA tasks. This breakthrough not only improves the accuracy of authorship attribution but also broadens its applicability to include regional linguistic analysis, making a substantial contribution to the field of computational linguistics.},
booktitle = {Companion Proceedings of the ACM on Web Conference 2024},
pages = {1657–1666},
numpages = {10},
keywords = {authorship attribution, disentanglement, regional dataset},
location = {Singapore, Singapore},
series = {WWW '24}
}
```

## Implementation
For the first Contrastive Stage, please run the script as the following:
'''
python examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --do_contrastive_cls \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 4 \
    --test_file data/AA_data/AA_cls_test.json \
    --validation_file data/AA_data/AA_cls_test.json \
    --train_file data/AA_data/AA_cls_train.json \
    --output_dir AA_region_cls/ \
    --overwrite_output_dir \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=32 \
    --save_strategy no \
    --evaluation_strategy epoch
'''

For the Disentanglement Stage, please run the script as the following:
python src/transformer/train.py 


## Evaluation 
For the ranking part, you can use the following command:
'''
python ranking_eva.py
'''
