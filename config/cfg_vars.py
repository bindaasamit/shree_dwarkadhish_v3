#----------------------------------------------------------------------------------------------------------------------
#                                               FILE PATHS
#----------------------------------------------------------------------------------------------------------------------
historical_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/input/historical_data/'
nifty_file = 'D:/myCourses/shree_dwarkadhish_v3/data/input/nifty.xlsx'
input_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/input/daily_stocks_data/'

sdmodel_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/sd_model/'
rpmodel_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/rp_model/'
volatility_dir = 'D:/myCourses/shree_dwarkadhish_v5/data/output/results/volatility/'

emascalpmodel_dir = 'D:/myCourses/shree_dwarkadhish_v5/data/output/results/ema_scalp_model/'
swing_model_dir = 'D:/myCourses/shree_dwarkadhish_v5/data/output/results/swing_model/'
swing_model_dir1 = 'D:/myCourses/shree_dwarkadhish_v5/data/output/results/sm_supertrend/'
swing_model_dir2 = 'D:/myCourses/shree_dwarkadhish_v5/data/output/results/sm_breakout/'
swing_vcp_model_dir = 'D:/myCourses/shree_dwarkadhish_v5/data/output/results/sm_vcp/'
swing_reversal_rsi_model_dir =  'D:/myCourses/shree_dwarkadhish_v5/data/output/results/sm_reversal_rsi/'

weekly_data_dir= 'D:/myCourses/shree_dwarkadhish_v5/data/output/results/weekly_data/'
db_dir = 'D:/myCourses/shree_dwarkadhish_v5/data/db/'


master_series_expiry_daterange_summary_path = 'D:/myCourses/shree_dwarkadhish_v3/data/input/expiry_data/master_series_expiry_daterange_summary.xlsx'
#volatility_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/input/expiry_data/2002_2025/'
hist_close_price_file_path = 'D:/myCourses/shree_dwarkadhish_v3/data/input/historical_closing_prices/historical_fno_closing_prices.csv'

json_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/input/json/'
ema_results_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/scanner/ema/'
ich_results_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/scanner/ich/'
results_backtest_dir = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/backtest_input/'
scanner_log_path = "D:/myCourses/shree_dwarkadhish_v3/data/output/logs/stock_recommender.log"
backtester_log_path = "D:/myCourses/shree_dwarkadhish_v3/data/output/logs/backtrader.log"

backtest_ema_path = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/backtest/ema/'
backtest_ich_path = 'D:/myCourses/shree_dwarkadhish_v3/data/output/results/backtest/ich/'

## Database and Table Names
db_name = "shree_dwarkadhish.db"
inp_table_name = "historical_stocks"
output_table_name = "signal_results"

hist_vol_range_data = [
['series_2002_03','2002-03-01','2002-03-28',21],
['series_2002_04','2002-03-29','2002-04-25',25],
['series_2002_05','2002-04-26','2002-05-30',21],
['series_2002_06','2002-05-31','2002-06-27',24],
['series_2002_07','2002-06-28','2002-07-25',25],
['series_2002_08','2002-07-26','2002-08-28',22],
['series_2002_09','2002-08-29','2002-09-26',24],
['series_2002_10','2002-09-27','2002-10-31',20],
['series_2002_11','2002-11-01','2002-11-28',24],
['series_2002_12','2002-11-29','2002-12-26',25],
['series_2003_01','2002-12-27','2003-01-30',20],
['series_2003_02','2003-01-31','2003-02-27',21],
['series_2003_03','2003-02-28','2003-03-27',22],
['series_2003_04','2003-03-28','2003-04-24',24],
['series_2003_05','2003-04-25','2003-05-29',22],
['series_2003_06','2003-05-30','2003-06-26',25],
['series_2003_07','2003-06-27','2003-07-31',19],
['series_2003_08','2003-08-01','2003-08-27',23],
['series_2003_09','2003-08-28','2003-09-25',25],
['series_2003_10','2003-09-26','2003-10-30',20],
['series_2003_11','2003-10-31','2003-11-27',23],
['series_2003_12','2003-11-28','2003-12-24',26],
['series_2004_01','2003-12-25','2004-01-29',20],
['series_2004_02','2004-01-30','2004-02-26',23],
['series_2004_03','2004-02-27','2004-03-25',24],
['series_2004_04','2004-03-26','2004-04-29',22],
['series_2004_05','2004-04-30','2004-05-27',24],
['series_2004_06','2004-05-28','2004-06-24',26],
['series_2004_07','2004-06-25','2004-07-29',23],
['series_2004_08','2004-07-30','2004-08-26',25],
['series_2004_09','2004-08-27','2004-09-30',20],
['series_2004_10','2004-10-01','2004-10-28',21],
['series_2004_11','2004-10-29','2004-11-25',26],
['series_2004_12','2004-11-26','2004-12-30',25],
['series_2005_01','2004-12-31','2005-01-27',22],
['series_2005_02','2005-01-28','2005-02-24',24],
['series_2005_03','2005-02-25','2005-03-31',20],
['series_2005_04','2005-04-01','2005-04-28',23],
['series_2005_05','2005-04-29','2005-05-26',25],
['series_2005_06','2005-05-27','2005-06-30',21],
['series_2005_07','2005-07-01','2005-07-27',23],
['series_2005_08','2005-07-28','2005-08-25',25],
['series_2005_09','2005-08-26','2005-09-29',21],
['series_2005_10','2005-09-30','2005-10-27',22],
['series_2005_11','2005-10-28','2005-11-24',26],
['series_2005_12','2005-11-25','2005-12-29',20],
['series_2006_01','2005-12-30','2006-01-25',22],
['series_2006_02','2006-01-26','2006-02-23',25],
['series_2006_03','2006-02-24','2006-03-30',18],
['series_2006_04','2006-03-31','2006-04-27',23],
['series_2006_05','2006-04-28','2006-05-25',26],
['series_2006_06','2006-05-26','2006-06-29',22],
['series_2006_07','2006-06-30','2006-07-27',24],
['series_2006_08','2006-07-28','2006-08-31',22],
['series_2006_09','2006-09-01','2006-09-28',20],
['series_2006_10','2006-09-29','2006-10-26',25],
['series_2006_11','2006-10-27','2006-11-30',20],
['series_2006_12','2006-12-01','2006-12-28',21],
['series_2007_01','2006-12-29','2007-01-25',21],
['series_2007_02','2007-01-26','2007-02-22',25],
['series_2007_03','2007-02-23','2007-03-29',21],
['series_2007_04','2007-03-30','2007-04-26',23],
['series_2007_05','2007-04-27','2007-05-31',26],
['series_2007_06','2007-06-01','2007-06-28',23],
['series_2007_07','2007-06-29','2007-07-26',25],
['series_2007_08','2007-07-27','2007-08-30',21],
['series_2007_09','2007-08-31','2007-09-27',23],
['series_2007_10','2007-09-28','2007-10-25',26],
['series_2007_11','2007-10-26','2007-11-29',20],
['series_2007_12','2007-11-30','2007-12-27',24],
['series_2008_01','2007-12-28','2008-01-31',26],
['series_2008_02','2008-02-01','2008-02-28',19],
['series_2008_03','2008-02-29','2008-03-27',22],
['series_2008_04','2008-03-28','2008-04-24',24],
['series_2008_05','2008-04-25','2008-05-29',22],
['series_2008_06','2008-05-30','2008-06-26',25],
['series_2008_07','2008-06-27','2008-07-31',25],
['series_2008_08','2008-08-01','2008-08-28',22],
['series_2008_09','2008-08-29','2008-09-25',23],
['series_2008_10','2008-09-26','2008-10-29',20],
['series_2008_11','2008-10-30','2008-11-26',22],
['series_2008_12','2008-11-27','2008-12-24',24],
['series_2009_01','2008-12-25','2009-01-29',20],
['series_2009_02','2009-01-30','2009-02-26',21],
['series_2009_03','2009-02-27','2009-03-26',21],
['series_2009_04','2009-03-27','2009-04-29',20],
['series_2009_05','2009-04-30','2009-05-28',23],
['series_2009_06','2009-05-29','2009-06-25',26],
['series_2009_07','2009-06-26','2009-07-30',22],
['series_2009_08','2009-07-31','2009-08-27',22],
['series_2009_09','2009-08-28','2009-09-24',24],
['series_2009_10','2009-09-25','2009-10-29',21],
['series_2009_11','2009-10-30','2009-11-26',23],
['series_2009_12','2009-11-27','2009-12-31',19],
['series_2010_01','2010-01-01','2010-01-28',20],
['series_2010_02','2010-01-29','2010-02-25',22],
['series_2010_03','2010-02-26','2010-03-25',24],
['series_2010_04','2010-03-26','2010-04-29',22],
['series_2010_05','2010-04-30','2010-05-27',24],
['series_2010_06','2010-05-28','2010-06-24',26],
['series_2010_07','2010-06-25','2010-07-29',23],
['series_2010_08','2010-07-30','2010-08-26',24],
['series_2010_09','2010-08-27','2010-09-30',21],
['series_2010_10','2010-10-01','2010-10-28',22],
['series_2010_11','2010-10-29','2010-11-25',25],
['series_2010_12','2010-11-26','2010-12-30',21],
['series_2011_01','2010-12-31','2011-01-27',22],
['series_2011_02','2011-01-28','2011-02-24',24],
['series_2011_03','2011-02-25','2011-03-31',18],
['series_2011_04','2011-04-01','2011-04-28',23],
['series_2011_05','2011-04-29','2011-05-26',25],
['series_2011_06','2011-05-27','2011-06-30',21],
['series_2011_07','2011-07-01','2011-07-28',22],
['series_2011_08','2011-07-29','2011-08-25',25],
['series_2011_09','2011-08-26','2011-09-29',20],
['series_2011_10','2011-09-30','2011-10-26',22],
['series_2011_11','2011-10-27','2011-11-24',24],
['series_2011_12','2011-11-25','2011-12-29',22],
['series_2012_01','2011-12-30','2012-01-25',23],
['series_2012_02','2012-01-26','2012-02-23',25],
['series_2012_03','2012-02-24','2012-03-29',20],
['series_2012_04','2012-03-30','2012-04-26',24],
['series_2012_05','2012-04-27','2012-05-31',26],
['series_2012_06','2012-06-01','2012-06-28',23],
['series_2012_07','2012-06-29','2012-07-25',24],
['series_2012_08','2012-07-26','2012-08-30',20],
['series_2012_09','2012-08-31','2012-09-27',21],
['series_2012_10','2012-09-28','2012-10-25',23],
['series_2012_11','2012-10-26','2012-11-29',21],
['series_2012_12','2012-11-30','2012-12-27',24],
['series_2013_01','2012-12-28','2013-01-31',20],
['series_2013_02','2013-02-01','2013-02-28',19],
['series_2013_03','2013-03-01','2013-03-28',21],
['series_2013_04','2013-03-29','2013-04-25',25],
['series_2013_05','2013-04-26','2013-05-30',21],
['series_2013_06','2013-05-31','2013-06-27',24],
['series_2013_07','2013-06-28','2013-07-25',23],
['series_2013_08','2013-07-26','2013-08-29',21],
['series_2013_09','2013-08-30','2013-09-26',23],
['series_2013_10','2013-09-27','2013-10-31',20],
['series_2013_11','2013-11-01','2013-11-28',22],
['series_2013_12','2013-11-29','2013-12-26',26],
['series_2014_01','2013-12-27','2014-01-30',20],
['series_2014_02','2014-01-31','2014-02-26',21],
['series_2014_03','2014-02-27','2014-03-27',20],
['series_2014_04','2014-03-28','2014-04-23',25],
['series_2014_05','2014-04-24','2014-05-29',22],
['series_2014_06','2014-05-30','2014-06-26',24],
['series_2014_07','2014-06-27','2014-07-31',19],
['series_2014_08','2014-08-01','2014-08-28',23],
['series_2014_09','2014-08-29','2014-09-25',22],
['series_2014_10','2014-09-26','2014-10-30',19],
['series_2014_11','2014-10-31','2014-11-27',23],
['series_2014_12','2014-11-28','2014-12-24',26],
['series_2015_01','2014-12-25','2015-01-29',20],
['series_2015_02','2015-01-30','2015-02-26',22],
['series_2015_03','2015-02-27','2015-03-26',21],
['series_2015_04','2015-03-27','2015-04-30',19],
['series_2015_05','2015-05-01','2015-05-28',23],
['series_2015_06','2015-05-29','2015-06-25',26],
['series_2015_07','2015-06-26','2015-07-30',21],
['series_2015_08','2015-07-31','2015-08-27',22],
['series_2015_09','2015-08-28','2015-09-24',24],
['series_2015_10','2015-09-25','2015-10-29',19],
['series_2015_11','2015-10-30','2015-11-26',23],
['series_2015_12','2015-11-27','2015-12-31',25],
['series_2016_01','2016-01-01','2016-01-28',22],
['series_2016_02','2016-01-29','2016-02-25',22],
['series_2016_03','2016-02-26','2016-03-31',18],
['series_2016_04','2016-04-01','2016-04-28',23],
['series_2016_05','2016-04-29','2016-05-26',25],
['series_2016_06','2016-05-27','2016-06-30',20],
['series_2016_07','2016-07-01','2016-07-28',23],
['series_2016_08','2016-07-29','2016-08-25',24],
['series_2016_09','2016-08-26','2016-09-29',19],
['series_2016_10','2016-09-30','2016-10-27',23],
['series_2016_11','2016-10-28','2016-11-24',26],
['series_2016_12','2016-11-25','2016-12-29',21],
['series_2017_01','2016-12-30','2017-01-25',22],
['series_2017_02','2017-01-26','2017-02-23',25],
['series_2017_03','2017-02-24','2017-03-30',19],
['series_2017_04','2017-03-31','2017-04-27',23],
['series_2017_05','2017-04-28','2017-05-25',25],
['series_2017_06','2017-05-26','2017-06-29',22],
['series_2017_07','2017-06-30','2017-07-27',23],
['series_2017_08','2017-07-28','2017-08-31',21],
['series_2017_09','2017-09-01','2017-09-28',20],
['series_2017_10','2017-09-29','2017-10-26',25],
['series_2017_11','2017-10-27','2017-11-30',20],
['series_2017_12','2017-12-01','2017-12-28',22],
['series_2018_01','2017-12-29','2018-01-25',23],
['series_2018_02','2018-01-26','2018-02-22',23],
['series_2018_03','2018-02-23','2018-03-28',21],
['series_2018_04','2018-03-29','2018-04-26',24],
['series_2018_05','2018-04-27','2018-05-31',21],
['series_2018_06','2018-06-01','2018-06-28',23],
['series_2018_07','2018-06-29','2018-07-26',24],
['series_2018_08','2018-07-27','2018-08-30',19],
['series_2018_09','2018-08-31','2018-09-27',22],
['series_2018_10','2018-09-28','2018-10-25',20],
['series_2018_11','2018-10-26','2018-11-29',21],
['series_2018_12','2018-11-30','2018-12-27',23],
['series_2019_01','2018-12-28','2019-01-31',20],
['series_2019_02','2019-02-01','2019-02-28',19],
['series_2019_03','2019-03-01','2019-03-28',21],
['series_2019_04','2019-03-29','2019-04-25',25],
['series_2019_05','2019-04-26','2019-05-30',20],
['series_2019_06','2019-05-31','2019-06-27',24],
['series_2019_07','2019-06-28','2019-07-25',24],
['series_2019_08','2019-07-26','2019-08-29',20],
['series_2019_09','2019-08-30','2019-09-26',22],
['series_2019_10','2019-09-27','2019-10-31',20],
['series_2019_11','2019-11-01','2019-11-28',22],
['series_2019_12','2019-11-29','2019-12-26',25],
['series_2020_01','2019-12-27','2020-01-30',20],
['series_2020_02','2020-01-31','2020-02-27',21],
['series_2020_03','2020-02-28','2020-03-26',21],
['series_2020_04','2020-03-27','2020-04-30',24],
['series_2020_05','2020-05-01','2020-05-28',23],
['series_2020_06','2020-05-29','2020-06-25',26],
['series_2020_07','2020-06-26','2020-07-30',22],
['series_2020_08','2020-07-31','2020-08-27',24],
['series_2020_09','2020-08-28','2020-09-24',24],
['series_2020_10','2020-09-25','2020-10-29',22],
['series_2020_11','2020-10-30','2020-11-26',23],
['series_2020_12','2020-11-27','2020-12-31',20],
['series_2021_01','2021-01-01','2021-01-28',21],
['series_2021_02','2021-01-29','2021-02-25',22],
['series_2021_03','2021-02-26','2021-03-25',22],
['series_2021_04','2021-03-26','2021-04-29',21],
['series_2021_05','2021-04-30','2021-05-27',24],
['series_2021_06','2021-05-28','2021-06-24',25],
['series_2021_07','2021-06-25','2021-07-29',22],
['series_2021_08','2021-07-30','2021-08-26',24],
['series_2021_09','2021-08-27','2021-09-30',20],
['series_2021_10','2021-10-01','2021-10-28',21],
['series_2021_11','2021-10-29','2021-11-25',26],
['series_2021_11','2021-11-26','2021-12-30',21],
['series_2022_01','2021-12-31','2022-01-27',22],
['series_2022_02','2022-01-28','2022-02-24',24],
['series_2022_03','2022-02-25','2022-03-31',19],
['series_2022_04','2022-04-01','2022-04-28',22],
['series_2022_05','2022-04-29','2022-05-26',25],
['series_2022_06','2022-05-27','2022-06-30',21],
['series_2022_07','2022-07-01','2022-07-28',22],
['series_2022_08','2022-07-29','2022-08-25',25],
['series_2022_09','2022-08-26','2022-09-29',20],
['series_2022_10','2022-09-30','2022-10-27',23],
['series_2022_11','2022-10-28','2022-11-24',26],
['series_2022_12','2022-11-25','2022-12-29',22],
['series_2023_01','2022-12-30','2023-01-25',23],
['series_2023_02','2023-01-26','2023-02-23',24],
['series_2023_03','2023-02-24','2023-03-29',18],
['series_2023_04','2023-03-30','2023-04-27',23],
['series_2023_05','2023-04-28','2023-05-25',25],
['series_2023_06','2023-05-26','2023-06-28',22],
['series_2023_07','2023-06-29','2023-07-27',24],
['series_2023_08','2023-07-28','2023-08-31',20],
['series_2023_09','2023-09-01','2023-09-28',21],
['series_2023_10','2023-09-29','2023-10-26',23],
['series_2023_11','2023-10-27','2023-11-30',20],
['series_2023_12','2023-12-01','2023-12-28',23],
['series_2024_01','2023-12-29','2024-01-25',24],
['series_2024_02','2024-01-26','2024-02-29',20],
['series_2024_03','2024-03-01','2024-03-28',20],
['series_2024_04','2024-03-29','2024-04-25',24],
['series_2024_05','2024-04-26','2024-05-30',20],
['series_2024_06','2024-05-31','2024-06-27',23],
['series_2024_07','2024-06-28','2024-07-25',25],
['series_2024_08','2024-07-26','2024-08-29',22],
['series_2024_09','2024-08-30','2024-09-26',24],
['series_2024_10','2024-09-27','2024-10-31',20],
['series_2024_11','2024-11-01','2024-11-28',22],
['series_2024_12','2024-11-29','2024-12-26',26],
['series_2025_01','2024-12-27','2025-01-30',21],
['series_2025_02','2025-01-31','2025-02-27',20],
['series_2025_03','2025-02-28','2025-03-27',20],
['series_2025_04','2025-03-28','2025-04-24',25],
['series_2025_05','2025-04-25','2025-05-29',22],
['series_2025_06','2025-05-30','2025-06-26',25],
['series_2025_07','2025-06-27','2025-07-31',19],
['series_2025_08','2025-08-01','2025-08-28',23],
['series_2025_09','2025-08-29','2025-09-30',21],
['series_2025_10','2025-10-01','2025-10-28',22],
['series_2025_11','2025-10-29','2025-11-25',22],
['series_2025_12','2025-11-26','2025-12-30',25]
]

#----------------------------------------------------------------------------------------------------------------------
#                                               LOADING VARIABLES
#----------------------------------------------------------------------------------------------------------------------
hist_file_columns= ['tckr_symbol','trade_dt','scty_series','open_price',
                   'high_price','low_price','closing_price','last_price',
                   'prev_closing_price','total_trading_volume','total_transfer_value']

# required for incremental loading
required_columns = ['TckrSymb','TradDt', 'SctySrs', 'OpnPric', 'HghPric', 'LwPric', 'ClsPric', 'LastPric', 'PrvsClsgPric', 'TtlTradgVol', 'TtlTrfVal']
column_mapping = {
                'TradDt' : 'trade_dt',  
                'TckrSymb' : 'tckr_symbol',
                'SctySrs' : 'scty_series',  
                'OpnPric' : 'open_price',
                'HghPric' : 'high_price',
                'LwPric'  : 'low_price', 
                'ClsPric' : 'closing_price',     
                'LastPric': 'last_price',
                'PrvsClsgPric' : 'prev_closing_price',
                'TtlTradgVol' : 'total_trading_volume',
                'TtlTrfVal' : 'total_transfer_value'
            }
# Read Query
nse_data_read_query = "SELECT tckr_symbol, trade_dt, scty_series, open_price, high_price, low_price, closing_price,last_price, prev_closing_price, total_trading_volume, total_transfer_value from historical_stocks"
#----------------------------------------------------------------------------------------------------------------------
#                                               EMA VARIABLES
#----------------------------------------------------------------------------------------------------------------------

warning_indicators = ['ema_sideways_market', 
                      'ema_whipsaw_risk', 
                      'ema_ribbon_width']

only_signals = ['profit_n_loss_pct', 
                'signal_action', 
                'signal_warning' , 
                'signal_limiter', 
                'signal_reason', 
                'signal_confidence', 
                'signal_type'] 

bullish_indicators = ['ema_golden_cross_filtered',
                      'ema_golden_cross', 
                      'ema_slopes_aligned_buy', 
                      'rsi_bullish', 
                      'macd_bullish',
                      'adx_bullish', 
                      'bullish_volume',
                      'ema_trend_strength',
                      'ema_bullish_alignment', 
                      'ema_alignment_score', 
                      'ema_slope_5', 'ema_slope_13', 'ema_slope_26', 'ema_avg_slope']

bearish_indicators = ['ema_death_cross_filtered', 
                      'ema_death_cross', 
                      'ema_slopes_aligned_sell', 
                      'rsi_bearish', 
                      'macd_bearish',
                      'adx_bearish', 
                      'bearish_volume', 
                      'ema_trend_strength',
                      'ema_bearish_alignment', 
                      'ema_alignment_score', 
                      'ema_slope_5', 
                      'ema_slope_13', 
                      'ema_slope_26', 
                      'ema_avg_slope']

key_fields = ['tckr_symbol', 'trade_dt']

input_fields = ['scty_series', 'open_price', 'high_price', 'low_price', 
                'closing_price', 'last_price', 'prev_closing_price', 
                'total_trading_volume', 'total_transfer_value', 
                'ema5', 'ema13', 'ema26', 'rsi', 'adx', 'atr', 'volatility_regime']

ema_ordered_cols = key_fields + warning_indicators + only_signals + bullish_indicators + bearish_indicators + input_fields


###. EMA_BUY_VARS
ema_buy_vars = key_fields + warning_indicators + only_signals + bullish_indicators + input_fields
ema_sell_vars = key_fields + warning_indicators + only_signals + bearish_indicators + input_fields
ema_noaction_vars = ema_ordered_cols

round_ema_num_vars = {
    'profit_n_loss_pct':2, 'ema5':2, 'ema13':2, 
    'ema26':2, 'ema_ribbon_width':2, 'ema_ribbon_width_pct':0, 
    'ema_ribbon_avg':2, 'ema_slope_5':2, 'ema_slope_13':2, 
    'ema_slope_26':2, 'ema_avg_slope':2, 'ema_alignment_score':2, 
    'ema_trend_strength':2
}
#----------------------------------------------------------------------------------------------------------------------
#                                               ICHIMOKU VARIABLES
#----------------------------------------------------------------------------------------------------------------------

key_fields = ['tckr_symbol', 'trade_dt']

ich_warning_indicators = ['tradeable', 'trend_quality']


buy_key_cols = ['signal_action', 'buy_score', 'strong_buy_entry_point', 'bullish_streak_end','signal_reason', 'buy_reasons', 'signal_limiter']
sell_key_cols = ['signal_action', 'sell_score', 'strong_sell_entry_point', 'bearish_streak_end','signal_reason', 'sell_reasons', 'signal_limiter']
other_signal_cols = ['signal_warning','signal_confidence', 'signal_type'] 

buy_sell_key_cols =  ['signal_action', 'buy_score', 'strong_buy_entry_point','bullish_streak_end','sell_score', 'strong_sell_entry_point','bearish_streak_end','signal_reason', 'signal_limiter','buy_reasons', 'sell_reasons','signal_warning','signal_confidence', 'signal_type']



ich_bullish_indicators = ['kumo_twist_bullish_filtered','price_kumo_bullish_breakout_filtered', 
                          'tenkan_kijun_bullish_crossover_filtered','price_kijun_bullish_crossover_filtered']

ich_bearish_indicators = ['price_kumo_bearish_breakout_filtered', 'price_kumo_bearish_breakdown_filtered']

filtering_metrics = ['kumo_thickness', 'kumo_thickness_pct', 'tk_separation', 'tk_separation_pct',
                     'price_kumo_distance', 'price_kumo_distance_pct', 'chikou_clear', 'price_range_pct', 'avg_range_pct']



ich_input_fields = ['tenkan', 'kijun','senkou_a', 'senkou_b', 'chikou_span', 'kumo_top', 'kumo_bottom', 'prev_closing_price',
                    'scty_series', 'open_price', 'high_price', 'low_price', 'closing_price', 'last_price', 'prev_closing_price', 
                    'total_trading_volume', 'total_transfer_value']

ich_warning_indicators_additional = ['sideways_market', 'weak_trend', 'choppy_market', 'false_breakout_risk'] 

ich_ordered_cols = key_fields + ich_warning_indicators + buy_sell_key_cols + ich_bullish_indicators + ich_bearish_indicators + filtering_metrics + ich_input_fields + ich_warning_indicators_additional

ich_buy_vars = key_fields + ich_warning_indicators + buy_key_cols + other_signal_cols + ich_bullish_indicators + filtering_metrics + ich_input_fields + ich_warning_indicators_additional
ich_sell_vars = key_fields + ich_warning_indicators + sell_key_cols + other_signal_cols + ich_bearish_indicators + filtering_metrics + ich_input_fields + ich_warning_indicators_additional
ich_noaction_vars = ich_ordered_cols

round_ich_num_vars = {'price_kumo_distance' : 2, 'price_kumo_distance_pct' : 2, 'price_range_pct' : 2, 'avg_range_pct' : 2, 
                      'trend_quality' : 2, 'tenkan' : 2, 'kijun' : 2, 'senkou_a' : 2,	'senkou_b' : 2, 'chikou_span' : 2,
                      'kumo_top' : 2, 'kumo_bottom' : 2,	'kumo_thickness' : 2, 'kumo_thickness_pct' : 2,	
                      'tk_separation' : 2, 'tk_separation_pct' : 2, 'trend_quality' : 2}