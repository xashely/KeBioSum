Details can be found [here] (https://github.com/allenai/s2orc). 

To prepare the dataset, follow instructions below:
1. You must contact owners to get access to this dataset. 
2. Copy links from the `.sh` file they send into a file similar to `./src/s2orc/links.txt`
2. Run  `make_download_links.py` to make a file similar to `./src/s2orc/download_links.txt`
4. Run `download_s2orc_and_filter.py` to restrict data to medical publications and download. 