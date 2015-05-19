# Neural_Tensor_Network_For_KB_Completion v0.1
A Neural Tensor Network for Knowledge Base Completion as described in the paper "Reasoning With Neural Tensor Networks For Knowledge Base Completion" (http://nlp.stanford.edu/~socherr/SocherChenManningNg_NIPS2013.pdf by Richard Socher*, Danqi Chen*, Christopher D. Manning and Andrew Y. Ng. 
(http://www.socher.org/index.php/Main/ReasoningWithNeuralTensorNetworksForKnowledgeBaseCompletion)

- Original Matlab code: download via webarchive: https://web.archive.org/web/20140807011802/http://www-nlp.stanford.edu/~socherr/codeDeepDB.zip
- Python version: https://github.com/siddharth-agrawal/Neural-Tensor-Network

- Java Implementation: here,
  - based on ND4j (http://nd4j.org/) as computation libary for linear algebra (as alternative to numpy and scipy in Python)
  - based on LBFGS Optimization based on https://github.com/aria42/nlp-utils
  - jmatio for loading *.mat word embeddings
  
- Before running the code:
 - Run Configurations / Arguments / add "-Xmx2G" in the box at the bottom
 - Edit path to training data


Additional information / resources:
- Architecture of NTN: 
 - digital: [WE dimension=6, slice size=3] http://fs1.directupload.net/images/150519/5owgxnhp.png  
 - draft: http://fs2.directupload.net/images/150428/hjxlxzom.jpg
- Classdiagram: http://fs2.directupload.net/images/150519/hqjyguo6.png


Main classes:
- Run_NTN: start the project
- NTN: Neural Tensor Network with Cost/Loss function
- Data_Factory: responsible for loading, providing and managing data

Accuracy with data from Socher et. al 2013: Freebase: Wordbase

Roadmap / next steps:
- Code review
- Maven integration
- Integration in DL4j Framework
- Multilingual support
 - Multilingual training data: spanish and german tripples and word vectors
- Improvements and Experiments
