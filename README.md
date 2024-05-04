# Fabric-Composition-Extraction by Fine-Tuning DONUT on custom dataset
Github repo for Fine-Tuning DONUT on self collected dataset

- To fine-tune our model on a custom dataset, first clone the repo and then create a Hugging Face Image Dataset Apache Arrow object by creating the required environment by running:
[create_environment.sh](https://github.com/azhara001/Fabric-Composition-Extraction/blob/main/create_environment.sh)

- Next, run the following:
[dataset_prep.py](https://github.com/azhara001/Fabric-Composition-Extraction/blob/main/dataset_prep.py)

- To fine-tune the model on your own dataset, run the following:
[Fine_Tuning_DONUT.ipynb](https://github.com/azhara001/Fabric-Composition-Extraction/blob/main/Fine_Tuning_DONUT.ipynb)

  - Note: This notebook draws major inspiration from [NielsRogge Github Repo](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut)

- To test our trained model on a custom image from the wild using our own fine-tuned model, run the following:
[Inference_Script_Robustness.ipynb](https://github.com/azhara001/Fabric-Composition-Extraction/blob/main/Inference%20Script_Robustness.ipynb)

- Finally, to sneak a glimpse of our model architecture and backbone, refer to the below image:
![Model Architecture](model_architecture.png)

- For feedback, questions, or comments or in case of spotting an error, please send an email: [abdullah_azhar@berkeley.edu](mailto:abdullah_azhar@berkeley.edu)


Relevant Repositories for the project:
1. [https://github.com/isirollan/capstone-app](https://github.com/isirollan/capstone-app) for the front-end of the web-app
2. [https://github.com/erinengle76/capstone-running-inference](https://github.com/erinengle76/capstone-running-inference) for deploying the model on AWS SageMaker
