DAILY COMMODITY FORECAST UPDATE - GITHUB ACTIONS PACKAGE
========================================================

This package is for the final no-news commodity forecasting platform.
It contains the saved dissertation model files plus a daily_update.py script.

WHAT IT DOES
------------
Every day, GitHub Actions can:
1. Download recent public market data.
2. Rebuild the latest no-news features.
3. Load your saved dissertation .pkl models.
4. Create updated forecasts.
5. Send the forecasts to your AwardSpace PHP API.
6. Your MySQL database and website update.
7. Activepieces can then email the updated summary.

IMPORTANT
---------
This package does NOT retrain the 3-hour models.
It loads saved models and creates fresh predictions.
The first GitHub run may take 5-15 minutes because it installs Python packages and downloads market data.
It should not take 1-3 hours.

GITHUB SETUP
------------
1. Create a new private GitHub repository.
2. Upload all files/folders from this package.

Because some .pkl files are large, the normal GitHub web uploader may reject them.
If that happens, use GitHub Desktop or the git command line instead.
Each file is below 100 MB, so normal git can accept them.

3. Go to repository Settings -> Secrets and variables -> Actions -> New repository secret.
4. Create this secret:

   Name: PLATFORM_API_KEY
   Value: your API key from AwardSpace config.php

5. Optional secret if needed:

   Name: PLATFORM_FORECAST_API_URL
   Value: http://commodity.fin-corp.uk/api/ingest_forecast.php

6. Go to Actions -> Daily Commodity Forecast Update.
7. Click Run workflow to test manually.
8. Check your website and phpMyAdmin after it finishes.

SCHEDULE
--------
The workflow file is set to run daily at 08:00 UTC.
During UK summer time, this is 09:00 BST.
Set Activepieces email/report time after this, for example 09:30 UK time.

EXPECTED DATABASE FORECAST ROWS
-------------------------------
GOLD:   1,2,3,5,10,20,30,40,60
CRUDE:  1,2,3,5,10,20
NATGAS: 1,2,3,5,10,15,20,30,40,60

SECURITY
--------
Do not put the API key directly inside daily_update.py.
Use GitHub Secrets only.
After the demo/submission, change your API key in config.php and update the GitHub secret and Activepieces URL.
