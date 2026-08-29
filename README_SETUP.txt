DAILY COMMODITY FORECAST UPDATE - GITHUB ACTIONS PACKAGE


This package is for the final no-news commodity forecasting platform. Final news model were not deployed because of the news unstructured nature that can be unpredictable and cause significant changes to the results either positive or negative, in contrary to the structured nature of quantitative price historical data and statistics.

This repository contains the saved dissertation model files plus a daily_update.py script.

WHAT IT DOES
------------
Every day, GitHub Actions can:
1. Download recent public market data.
2. Rebuild the latest no-news features.
3. Load the saved dissertation .pkl models.
4. Create updated forecasts.
5. Send the forecasts to the AwardSpace PHP API.
6. MySQL database and website update.

IMPORTANT
---------
This package does NOT retrain the 3 to 9 hour models.
It loads saved models and creates fresh predictions.
The first GitHub run may take 5-15 minutes because it installs Python packages and downloads market data.
It should not take 1-3 hours.

GITHUB SETUP
------------
1. Created a new private GitHub repository.
2. Uploaded all files/folders from this package.

Because some .pkl files are large, the normal GitHub web uploader may reject them.
If that happens, GitHub Desktop or the git command line were used instead.
Each file is below 100 MB, so normal git can accept them.

3. Went to repository Settings -> Secrets and variables -> Actions -> New repository secret.
4. Created this secret:

   Name: PLATFORM_API_KEY
   Value: the API key from AwardSpace config.php file

5. Optional secret if needed:

   Name: PLATFORM_FORECAST_API_URL
   Value: http://commodity.fin-corp.uk/api/ingest_forecast.php

6. Went to Actions -> Daily Commodity Forecast Update.
7. Clicked Run workflow to test manually.
8. Checked the website and phpMyAdmin after it finishes.

SCHEDULE
--------
The workflow file is set to run daily at 08:00 UTC.
During UK summer time, this is 09:00 BST.

EXPECTED DATABASE FORECAST ROWS
-------------------------------
GOLD:   1,2,3,5,10,20,30,40,60
CRUDE:  1,2,3,5,10,20
NATGAS: 1,2,3,5,10,15,20,30,40,60

SECURITY
--------
the API key was not put directly inside daily_update.py.
Only GitHub Secrets has been used.
After the demo/submission, the API key in config.php was changed and GitHub secret was updated.
