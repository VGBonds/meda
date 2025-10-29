# МedА
### A medical assistant application

### Install requirements:
```bash
pip install -r requirements.txt
```
### Provide HF token:
add your Hugging Face API token in the .env file:

```bash
HUGGINGFACEHUB_API_TOKEN=your_hugging```
```

### Run the fine tuning script:
```bash
python src/peft_medgemma_4b_it_nih_pneumonia_chest_x_ray.py
```
### Run the test script:
```bash
python src/test_medgemma_49_it_nih_pneumonia_chest_x_ray
``` 

