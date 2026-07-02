# Word frequency calibration using external information and topic models

Code accompanying our poster at AMLaP 2026.

The main components of the repository are the following: 

```
locations.py        # Variables specifying paths where data can be found
prepare_data.ipynb  # Data preparation
analysis.ipynb      # Main analysis notebook
```

The rest of the files (PDFs and CSVs containing the results and figures) can be recreated using the analysis.ipynb notebook, provided that the data is available in the locations specified in locations.py.
The abstract itself is under ./AMLaP2026-LaTeX-Template/AMLaP2026-Template.pdf.

## Instructions

Follow these instructions to reproduce the results in the abstract, or create results for your own corpus. 

1. create a virtual (Python 3.12) environment, and install the dependencies: 

    ```
    pip install -r requirements.txt
    ```
2. Then check the locations.py file to check what variables are used to look up files required for the analysis. 
3. Use the notebook prepare_data.ipynb to download and install the needed datasets. The notebook contains more detailed instructions. 
4. Use the notebook analysis.ipynb to collect and visualize results. 


## Citation

> Hronský, R. & Keuleers, E. (2026). *Word frequency calibration using external information and topic models.* Poster presented at AMLaP 2026, Saarland University.

Bibtex: 
```bibtex
@misc{hronsky2026wordfreq,
  author    = {Hronsk\'{y}, Rastislav and Keuleers, Emmanuel},
  title     = {Word Frequency Calibration Using External Information and Topic Models},
  howpublished = {Poster presented at AMLaP 2026},
  year      = {2026},
  address   = {Saarland University, Saarbr\"{u}cken, Germany},
}
```
