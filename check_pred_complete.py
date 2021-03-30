import glob
import argparse

def main(m4=False, lags=None):
    if lags is None:
        lags = [5, 10, 15]
    else:
        lags = [lags]

    if m4:
        all_models = glob.glob("results/m4_*_test.csv")
        all_datasets = ['m4_hourly_H' + str(n) for n in range(1, 21)]
        all_datasets += ['m4_daily_D' + str(n) for n in range(1, 21)]
        all_datasets += ['m4_weekly_W' + str(n) for n in range(1, 21)]
        #all_datasets += ['m4_quaterly_Q' + str(n) for n in range(1, 21)]     too small even for lag 5
        all_datasets += ['m4_monthly_M' + str(n) for n in range(1, 21)]
    else:
        all_models = glob.glob("results/[!m][!4]*_test.csv")
        all_datasets = ['bike_total_rents', 'bike_registered', 'bike_temperature', 'AbnormalHeartbeat', 'CatsDogs', 'Cricket', 'EOGHorizontalSignal', 'EthanolConcentration', 'Mallat', 'Phoneme', 'PigAirwayPressure', 'Rock', "SNP500", "NASDAQ", "DJI", "NYSE", "RUSSELL", "Energy_RH1", "Energy_RH2", "Energy_T4", "Energy_T5", "CloudCoverage"]

    for lag in lags:
        print("---- MISSING LAG {} ----".format(lag))
        for ds_name in all_datasets:
            composite = "results/{}_lag{}_test.csv".format(ds_name, lag)
            if composite not in all_models:
                print("missing:", composite)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M4", action="store_const", const=True, default=False)
    parser.add_argument("--lag", action="store", type=int)
    args = parser.parse_args()
    main(m4=args.M4, lags=args.lag)
