#### description
- 'chinese_L-12_H-768_A-12' is the pretrained BERT model for [huggingface Transformers](https://github.com/huggingface/transformers). 
    - You can choose any open-source pretraining weights, but remember to modify the config in ../embedding_example.yaml
- 'tag_list.txt' is a tag list demo for multi-label classification. You are encouraged to choose your own supervised signal including tags, categories and etc.

#### contents in 'data' folder in tree-like format
```
├── tag_list.txt  
├── desc.json  
├── chinese_L-12_H-768_A-12  
│   ├── vocab.txt  
│   ├── bert_google.bin   
├── pairwise  
│   ├── label.tsv  
│   └── pairwise.tfrecords  
├── pointwise  
│   ├── pretrain_0.tfrecords  
│   ├── pretrain_10.tfrecords  
│   ├── pretrain_11.tfrecords  
│   ├── pretrain_12.tfrecords  
│   ├── pretrain_13.tfrecords  
│   ├── pretrain_14.tfrecords  
│   ├── pretrain_15.tfrecords  
│   ├── pretrain_16.tfrecords  
│   ├── pretrain_17.tfrecords  
│   ├── pretrain_18.tfrecords  
│   ├── pretrain_19.tfrecords  
│   ├── pretrain_1.tfrecords  
│   ├── pretrain_2.tfrecords  
│   ├── pretrain_3.tfrecords  
│   ├── pretrain_4.tfrecords  
│   ├── pretrain_5.tfrecords  
│   ├── pretrain_6.tfrecords  
│   ├── pretrain_7.tfrecords  
│   ├── pretrain_8.tfrecords  
│   └── pretrain_9.tfrecords  
└── test_a  
    └── test_a.tfrecords  
```