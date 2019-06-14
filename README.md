# CountrySpecs
Webscraping project putting the CIA worldfactbook data into machine readable format

This project was run using the Anaconda distribution on Windows 10.

## Installation and Setup
1. Install [Python](https://www.python.org/downloads/) and [Anaconda](https://www.anaconda.com/distribution/)

2. Open the Anaconda cmd prompt and look at its path. Move the countries.yml file to there.

3. Create a new environment using the countries.yml file by typing in: 

`conda env create -f countries.yml`

If the countries.yml is not in the right folder, then its full path must be specified:

`conda env create -f C:\folder1\folder2\...\countries.yml`

4. Activate the environment via:

`conda activate countries`

5. Open up Jupyter in that environment by typing in:

`python -m notebook`
