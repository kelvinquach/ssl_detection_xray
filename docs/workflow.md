\# Project Workflow



\## Topic



\*\*Nghiên cứu học bán giám sát cho dò tìm bất thường trên ảnh X-quang phổi với detector dựa trên attention/Transformer trong điều kiện hạn chế nhãn\*\*



\## Current phase



The project is currently in the dataset selection and metadata-only subset creation stage.



\## Dataset



VinBigData Chest X-ray Abnormalities Detection.



\## Current subset



The current subset is metadata-only:



\- 500 Normal / No Finding images

\- All abnormal images

\- No DICOM images are copied into the Git repository



\## Current completed outputs



```text

data/raw/vinbigdata/metadata\_subset

├── selected\_image\_ids.csv

├── subset\_train\_annotations.csv

├── abnormal\_image\_ids.csv

├── normal\_image\_ids\_500.csv

├── subset\_summary.csv

├── subset\_class\_distribution.csv

├── positive\_normal\_summary.csv

└── README\_metadata\_subset.md

