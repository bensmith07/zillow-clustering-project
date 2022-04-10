# Improving on the Zestimate
## Identifying drivers of error in Zillow's home value predictions

## About the project
We will conduct an in depth analysis of Zillow property data from 2017. We will use exploratory analysis and clustering techniques to identify the key drivers of error in Zillow's predictions, then use machine learning algorithms to create a model capable of predicting the error. 

### Project Description

Property values have skyrocketed over the last two years. With such rapid changes in home values, predicting those values has become even harder to predict than before. Since Zillow's estimate of home value is one of the primary drivers of website traffic, having a reliable estimate is paramount. Any improvement we can make on the previous model will help us out-estimate our competitors and keep us at the top as the most trusted name in real estate technology. 

This project will analyze property attributes in relation to the error in Zillow's 2017 prediction of the property's value, develop a model for the error based on those attributes, and leave with recommendations for how to improve future predictions. 

### Project Goals

By creating a reliable model for predicting property values, Zillow can enhance it's reputation for reliable property value estimates and better position itself in the real estate technology marketplace. 

### Initial Questions

- Which features are most highly correlated with the target?

- Are there clusters based on combination of bedroomcnt and bathroomcnt that are drivers of logerror?

- Are there clusters based on combination of bedroomcnt, bathroomcnt, and sqft that are drivers of logerror?

- Are there clusters based on combination of tax value per sqft, bedroomcnt, and bathroomcnt that are drivers of logerror?

- Are there clusters based on combination of latitude/longitude that are drivers of logerror?

### Data Dictionary

------------------
| Variable | Meaning 
|'airconditioningtypeid'	| Type of cooling system present in the home (if any)
|'architecturalstyletypeid'	| Architectural style of the home (i.e. ranch, colonial, split-level, etc…)
|'basementsqft'	|  Finished living area below or partially below ground level
|'bathroomcnt'	|  Number of bathrooms in home including fractional bathrooms
|'bedroomcnt' | 	 Number of bedrooms in home 
|'buildingqualitytypeid'	| Overall assessment of condition of the building from best (lowest) to worst (highest)
|'buildingclasstypeid'	|The building framing type (steel frame, wood frame, concrete/brick) 
|'calculatedbathnbr'	| Number of bathrooms in home including fractional bathroom
|'decktypeid' |	Type of deck (if any) present on parcel
|'threequarterbathnbr'	|  Number of 3/4 bathrooms in house (shower + sink + toilet)
|'finishedfloor1squarefeet' |	 Size of the finished living area on the first (entry) floor of the home
|'calculatedfinishedsquarefeet'	| Calculated total finished living area of the home 
|'finishedsquarefeet6' |	Base unfinished and finished area
|'finishedsquarefeet12' |	Finished living area
|'finishedsquarefeet13' |	Perimeter  living area
|'finishedsquarefeet15' |	Total area
|'finishedsquarefeet50'	| Size of the finished living area on the first (entry) floor of the home
|'fips'	| Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details
|'fireplacecnt'	| Number of fireplaces in a home (if any)
|'fireplaceflag'	| Is a fireplace present in this home 
|'fullbathcnt' |	 Number of full bathrooms (sink, shower + bathtub, and toilet) present in home
|'garagecarcnt' |	 Total number of garages on the lot including an attached garage
|'garagetotalsqft' |	 Total number of square feet of all garages on lot including an attached garage
|'hashottuborspa'	|  Does the home have a hot tub or spa
|'heatingorsystemtypeid' |	 Type of home heating system
|'latitude' |	 Latitude of the middle of the parcel multiplied by 10e6
|'longitude' |	 Longitude of the middle of the parcel multiplied by 10e6
|'lotsizesquarefeet' |	 Area of the lot in square feet
|'numberofstories' |	 Number of stories or levels the home has
|'parcelid' |	 Unique identifier for parcels (lots) 
|'poolcnt' |	 Number of pools on the lot (if any)
|'propertylandusetypeid'	 | Type of land use the property is zoned for
|'propertyzoningdesc' |	 Description of the allowed land uses (zoning) for that property
|'rawcensustractandblock' |	 Census tract and block ID combined - also contains blockgroup assignment by extension
|'censustractandblock'	 | Census tract and block ID combined - also contains blockgroup assignment by extension
|'regionidcounty' |	County in which the property is located
|'regionidcity'	 | City in which the property is located (if any)
|'regionidzip'	| Zip code in which the property is located
|'regionidneighborhood' |	Neighborhood in which the property is located
|'roomcnt' |	 Total number of rooms in the principal residence
|'storytypeid' |	 Type of floors in a multi-story house (i.e. basement and main level, split-level, attic, etc.).  See tab for details.
|'typeconstructiontypeid' |	 What type of construction material was used to construct the home
|'unitcnt' |	 Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...)
|'yardbuildingsqft17' |	Patio in  yard
|'yardbuildingsqft26' |	Storage shed/building in yard
|'yearbuilt' |	 The Year the principal residence was built 
|'taxvaluedollarcnt' |	The total tax assessed value of the parcel
|'structuretaxvaluedollarcnt' |	The assessed value of the built structure on the parcel
|'landtaxvaluedollarcnt' |	The assessed value of the land area of the parcel
|'taxamount' |	The total property tax assessed for that assessment year
|'assessmentyear' |	The year of the property tax assessment 
|'taxdelinquencyflag' |	Property taxes for this parcel are past due as of 2015
|'taxdelinquencyyear' |	Year for which the unpaid propert taxes were due 
------------------

### Steps to Reproduce

1. You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the zillow database. The env.py should also contain a function named get_db_url() that establishes the string value of the database url. Store that env file locally in the repository. 
2. clone my repo (including the acquire.py, prepare.py, explore.py, and model.py modules) (confirm .gitignore is hiding your env.py file)
3. libraries used are pandas, matplotlib, seaborn, numpy, sklearn, math.

### The Plan

1. Acquisition
- In this stage, we obtained Zillow 2017 property data by querying the Codeup MySQL database hosted at data.codeup.com. The original source of this data was the Zillow competition hosted by Kaggle.com
2. Preparation
- we cleaned and prepped the data by creating a function that : 
    - filters the data for only Single Family Residential properties
    - drops redundant foreign key identification code columns
    - drops other redundant, irrelevant, or non-useful columns
    - handles null values by:
        - filling null values with 0's in the following columns, since it is reasonable to assume nulls in these columns represent zero values: 
            - `fireplacecnt`, `garagecarcnt`, `garagetotalsqft`, `hashottuborspa`, `poolcnt`, `threequarterbathnbr`, `taxdelinquencyflag`
        - then dropping columns that remain where greater than 5% of values in that column are null
        - then dropping rows that remain with any null values
        - changes data types to more appropriately reflect the data they represent
    - adds the following engineered feature columns (see data-dictionary for details):
        - `age`, `bool_has_garage`, `bool_has_pool`, `bool_has_fireplace`, `taxvalue_per_sqft`, `taxvalue_per_bedroom`, `taxvalue_per_bathroom`
    - adds the following target-related columns (for exploration) (see data-dictionary for details): 
        - `abs_logerror`, `logerror_direction`
3. Exploration
- We conducted an initial exploration of the data by examining correlations between each of the potential features and the target
- Then explored further by using K-means clustering to engineer additional features which were found to be drivers of the target
4. Modeling
- Using the initial features that were most highly correlated with the target, then varying which engineered cluster features were added, we created various Ordinary Least Squares (OLS) Regression and Polynomial Regression models. 
_ We then chose the model which performed with the smallest error on unseen data.

### How did we do?

-------------------


-------------------

### Key Findings:

We found the following combinations of features created clusters that proved to be drivers of logerror (however minor):
- bedroomcnt, bathroomcnt
- bedroomcnt, bathroomcnt, sqft
- bedroomcnt, bathroomcnt, taxvalue_persqft
- latitude, longitude

### Recommendations:

While we did find features that are drivers of logerror, the effect size was minor, so I would not recommend using the models created here as the zole basis for improving the Zestimate. However, clustering was shown to be a useful exercise, so additional clustering exploration is recommended in an attempt to find clusters that serve as larger drivers. 

### Next Steps:

Given more time, we should examine additional combinations of features that might create useful clusters. Additionally, we should test additional numbers of K-means clusters for each combination of features that were used in this report. 

Additionally, since real estate markets are based heavily on location, I might expect models to perform better which individually focus on a distinct geographic area. So, we could create a different model for each of our latitude/longitude clusters, or we could further narrow the geographic focus by using zip code or neighborhood information or by creating a larger number of latitude/longitude clusters. 