# Breast Cancer Masking



### Installation

1. **Clone the Repository**
```
git clone [repository URL]
```



#### 2. Setup Environment
- **Windows Users**: 
  - Run `.\setup.bat` by double-clicking it or through the command line.
- **macOS/Linux Users**: 
  - Rename the file to setup.sh
  - In the terminal, navigate to the project directory and run: `sh setup.sh`


### Using the Virtual Environment

#### To Activate the Virtual Environment:
- **Windows**: 
  - Run `breast_cancer_masking_venv\Scripts\activate` in the command prompt.
- **macOS/Linux**: 
  - Run `source breast_cancer_masking_venv/bin/activate` in the terminal.

#### To Deactivate:
- Run `deactivate` in the command prompt or terminal.

### Updating Dependencies

- To update `requirements.txt` with new dependencies, first ensure you are in the virtual environment and then run:
```
pip freeze > requirements.txt
```

- To install any new dependencies, run:
```
pip install requirements.txt
```