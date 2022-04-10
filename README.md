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

### Key Findings:

### Recommendations:

### Next Steps: