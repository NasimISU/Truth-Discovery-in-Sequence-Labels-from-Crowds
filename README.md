<html>
<head>This is the repository for the paper: Truth Discovery in Sequential Labels from Crowds</head>

<h2> Requirements </h2>
 <ul>
  <li>tensorflow</li>
  <li>keras</li>
  <li>numpy</li>
  <li>shutil</li>
  <li>sklearn</li>
</ul> 

<h2>Folder Description</h2>
 <ol>
  <li><strong>crf-ma-datasets:</strong> Download the original crowdsourced dataset annotated by Rodrigues et. al from http://amilab.dei.uc.pt/fmpr/crf-ma-datasets.tar.gz. </li>
  <li><strong>pre_trained_bert:</strong> Download the pre-trained BERT model and place it into this folder.</li>
  <li><strong>NER:</strong> This folder contains processed dataset. NER/original_data folder contains processed crf-ma-datasets and NER/processed_test_data folder contains processed test set.</li>
  <li><strong>execution:</strong> All the execution results are stored in this folder. execution/calculations folder contains iteration wise results.</li>
</ol> 

<h2>Dataset details</h2>
<ol>
    <li>NER: Dataset can be found at http://amilab.dei.uc.pt/fmpr/crf-ma-datasets.tar.gz</li>
    <li>PICO: Dataset can be found at https://github.com/yinfeiy/PICO-data</li>
</ol>

<h2>System Description</h2>
All executions are done on Macbook Pro with 2.6 GHz 6-CoreIntel Core i7 processor and 16 GB memory.

<h2>Results</h2>

 <table style="width:100%">
  <tr>
    <th>Dataset</th>
    <th>Precision(S)</th>
    <th>Recall(S)</th>
    <th>F1(S)</th>
    <th>Precision(R)</th>
    <th>Recall(R)</th>
    <th>F1(R)</th>
  </tr>
  <tr>
    <td>NER</td>
    <td>83.02</td>
    <td>78.69</td>
    <td>80.79</td>
    <td>92.64</td>
    <td>92.47</td>
    <td>91.63</td>
  </tr>
  <tr>
    <td>PICO</td>
    <td>64.03</td>
    <td>52.62</td>
    <td>57.77</td>
    <td>92.20</td>
    <td>95.15</td>
    <td>93.65</td>
  </tr>
</table> 
NOTE: S refers to strict metrics and R refers to relaxed metrics.
<h2>Results reproduction commands</h2>
<ol>
    <li>original data preprocessing: python data_preprocessing.py</li>
    <li>execution data preprocessing: python execution_data_preprocessing.py</li>
    <li>execution: python execution.py</li>
    <li>result calculation: python calculations.py</li>
</ol>

NOTE 1: The execution files will be stored in execution folder and the result calculation will be stored in execution/calculations folder. We use conlleval evaluation scripts from  http://amilab.dei.uc.pt/fmpr/ma-crf.tar.gz.

NOTE 2: PICO results can be reproduced by executing the scripts on PICO dataset.

NOTE 3: We have provided the execution run results for each iteration in execution/calculations folder.

<h2>References</h2>
<ol> <li>CRF-MA: 
@article{rodrigues2014sequence,
  title={Sequence labeling with multiple annotators},
  author={Rodrigues, Filipe and Pereira, Francisco and Ribeiro, Bernardete},
  journal={Machine learning},
  volume={95},
  number={2},
  pages={165--181},
  year={2014},
  publisher={Springer}
}</li>
 <li>AggSLC: {@INPROCEEDINGS{9679072,
  author={Sabetpour, Nasim and Kulkarni, Adithya and Xie, Sihong and Li, Qi},
  booktitle={2021 IEEE International Conference on Data Mining (ICDM)}, 
  title={Truth Discovery in Sequence Labels from Crowds}, 
  year={2021},
  pages={539-548},
  doi={10.1109/ICDM51629.2021.00065}
  }<li>
<li>DL-CL: @inproceedings{rodrigues2018deep,
  title={Deep learning from crowds},
  author={Rodrigues, Filipe and Pereira, Francisco C},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}</li>
<li>BERT Pre trained: @article{devlin2018bert,
  title={Bert: Pre-training of deep bidirectional transformers for language understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={Proceedings of NAACL-HLT 2019, Association for Computational Linguistics},
  year={2019},
  pages = {4171–4186}
 }</li> 
<li>OPTSLA: @inproceedings{sabetpour-etal-2020-optsla,
    title = "{O}pt{SLA}: an Optimization-Based Approach for Sequential Label Aggregation",
    author = "Sabetpour, Nasim  and
      Kulkarni, Adithya  and
      Li, Qi",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.119",
    doi = "10.18653/v1/2020.findings-emnlp.119",
    pages = "1335--1340"}
</li>
</ol>


</html>


