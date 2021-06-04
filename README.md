# MPC-BERT: A Pre-Trained Language Model for Multi-Party Conversation Understanding
This repository contains the source code for the _ACL 2021_ paper [MPC-BERT: A Pre-Trained Language Model for Multi-Party Conversation Understanding](https://arxiv.org/pdf/2106.01541.pdf). Jia-Chen Gu, Chongyang Tao, Zhen-Hua Ling, Can Xu, Xiubo Geng, Daxin Jiang. <br>
Hopefully, preprint and code will be released at the beginning of June. Thanks for your patience. <br>

## Introduction
Recently, various neural models for multi-party conversation (MPC) have achieved impressive improvements on a variety of tasks such as addressee recognition, speaker identification and response prediction. 
However, these existing methods on MPC usually represent interlocutors and utterances individually and ignore the inherent complicated structure in MPC which may provide crucial interlocutor and utterance semantics and would enhance the conversation understanding process. 
To this end, we present MPC-BERT, a pre-trained model for MPC understanding that considers learning who says what to whom in a unified model with several elaborated self-supervised tasks. 
Particularly, these  tasks can be generally categorized into (1) interlocutor structure modeling including reply-to utterance recognition, identical speaker searching and pointer consistency distinction, and (2) utterance semantics modeling including masked shared utterance restoration and shared node detection. 
We evaluate MPC-BERT on three downstream tasks including addressee recognition, speaker identification and response selection. 
Experimental results show that MPC-BERT outperforms previous methods by large margins and achieves new state-of-the-art performance on all three downstream tasks at two benchmarks.

<div align=center><img src="image/result_ addressee_recognition.png" width=80%></div>

<div align=center><img src="image/result_speaker_identification.png" width=80%></div>

<div align=center><img src="image/result_response_selection.png" width=80%></div>


## Cite
If you think our work is helpful or use the code, please cite the following paper:
**"MPC-BERT: A Pre-Trained Language Model for Multi-Party Conversation Understanding"**
Jia-Chen Gu, Chongyang Tao, Zhen-Hua Ling, Can Xu, Xiubo Geng, Daxin Jiang. _ACL (2021)_

```
@inproceedings{gu-etal-2021-mpcbert,
 title = "MPC-BERT: A Pre-Trained Language Model for Multi-Party Conversation Understanding",
 author = "Gu, Jia-Chen  and
           Tao, Chongyang  and
           Ling, Zhen-Hua  and
           Xu, Can  and
           Geng, Xiubo  and
           Jiang, Daxin",
 booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP)",
 month = aug,
 year = "2021",
 address = "Online",
 publisher = "Association for Computational Linguistics",
 url = "https://www.aclweb.org/anthology/2021.acl-main.1881",
}
```


## Update
Please keep an eye on this repository if you are interested in our work.
Feel free to contact us (gujc@mail.ustc.edu.cn) or open issues.
