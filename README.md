# Large Area AFM Analysis and Control

This repository provides a toolkit for the analysis and control of large-area atomic force microscopy (AFM) data. The control features are specifically designed for the DriveAFM microscope from Nanosurf, including a GUI application for AFM image acquisition and stage control. However, the analysis pipeline is versatile and can be used with data from other AFM systems, making it suitable for a broad range of AFM equipment and applications.

## Repository Structure

The root directory contains two main folders:

- **`Analysis Large-Area AFM`**: Contains the primary analysis tools and libraries for processing AFM datasets. The folder includes:
  - **Analysis Notebook**:
    - **`Analysis.ipynb`**: A Jupyter Notebook that guides users through the analysis workflow. This notebook utilizes functions from the Python scripts in this folder to perform tasks such as stitching, flattening, segmentation, and data export on large-area AFM images.
  - **Python Scripts**: The following scripts serve as modular functions for different stages of the analysis:
    - **`batchprocess.py`**: Manages batch processing of AFM files, automating the application of processing steps to large datasets.
    - **`flattening_v2.py`**: Provides functions for flattening AFM images to correct background variations and ensure consistent baselines across images.
    - **`managefile.py`**: Handles file input and output operations, including loading, saving, and organizing data in structured formats such as HDF5, ideal for large-scale AFM data handling.
    - **`segmentation_v3.py`**: Contains segmentation tools to detect structures in AFM images, using a pre-trained YOLO model for identifying and segmenting specific regions (e.g., biofilms or bacteria). It also includes functions for extracting and summarizing statistics from segmented regions.
    - **`stitching_v2.py`**: Provides methods for image stitching, enabling the combination of multiple AFM tiles into a single, large-area image. This includes both coordinate-based and feature-based approaches for seamless integration.

- **`AppGridWithStage`**: A GUI application designed for controlling the DriveAFM microscope from Nanosurf. This application allows users to:
  - Configure and execute grid scanning with precise stage movement.
  - Automate image acquisition for large-area scanning.
  - Integrate with the analysis pipeline for streamlined data collection and processing.

### Output

Processed results from the `Analysis Large-Area AFM` notebook and scripts are saved in structured folders generated within the output directory. These may include:

- **Stitched Images**: Large, composite images formed from individual AFM tiles.
- **Flattened Data**: Baseline-corrected images with consistent height normalization.
- **Segmentation Masks and Statistics**: Generated masks and comprehensive statistics (e.g., area, orientation) for each identified region, facilitating detailed analysis.

## Usage

1. **Installation**: Clone the repository and install the required packages listed in `requirements.txt`:

    ```bash
    git clone https://github.com/Rmillansol/Large-Area-AFM-Analysis-and-Control
 
    pip install -r requirements.txt
    ```

2. **Running the Analysis**: Navigate to `Analysis Large-Area AFM` and open `Analysis.ipynb` in Jupyter Notebook to run the complete analysis pipeline. Follow the steps in the notebook to process AFM datasets from stitching through segmentation and data export.

3. **Using the GUI Application**: The `AppGridWithStage` folder contains a GUI for controlling the AFM microscope. Launch the application to configure grid scans, control stage movement, and acquire images. The acquired data can then be processed using the `Analysis Large-Area AFM` pipeline.

## Citation

If you use this repository for research purposes, please cite the following article:

> DOI: [10.21203/rs.3.rs-5537963/v1](https://doi.org/10.21203/rs.3.rs-5537963/v1)

Proper citation allows us to maintain and further develop this toolkit for the research community.

## Author

Developed by Ruben Millan-Solsona. For inquiries, contact at [solsonarm@ornl.gov](mailto:solsonarm@ornl.gov).
[Google Scholar](https://scholar.google.com/citations?hl=es&user=zEOJb2cAAAAJ) | [ORCID](https://orcid.org/0000-0003-0912-7246) 

## Related Work

For more details on the methodology and applications of this toolkit, see our related research article:
[Link to Article](https://www.researchsquare.com/article/rs-5537963/v1)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.