import glob
import argparse


def main(m4=False, lag=None):
    if lag is None:
        lag = [5, 10, 15]
    else:
        lag = [lag]

    if m4:
        all_models = glob.glob("models/*/m4_*.pth")
        all_datasets = ['m4_hourly_H' + str(n) for n in range(1, 21)]
        all_datasets += ['m4_daily_D' + str(n) for n in range(1, 21)]
        all_datasets += ['m4_weekly_W' + str(n) for n in range(1, 21)]
        all_datasets += ['m4_quaterly_Q' + str(n) for n in range(1, 21)]
        all_datasets += ['m4_monthly_M' + str(n) for n in range(1, 21)]
    else:
        all_models = glob.glob("models/*/[!m][!4]*.pth")
        all_datasets = ['bike_total_rents', 'bike_registered', 'bike_temperature', 'AbnormalHeartbeat', 'CatsDogs', 'Cricket', 'EOGHorizontalSignal', 'EthanolConcentration', 'Mallat', 'Phoneme', 'PigAirwayPressure', 'Rock', "SNP500", "NASDAQ", "DJI", "NYSE", "RUSSELL", "Energy_RH1", "Energy_RH2", "Energy_T4", "Energy_T5", "CloudCoverage"]

    for lag in [5, 10, 15]:
        print("---- MISSING LAG {} ----".format(lag))
        for model_name in ['as01_a','as01_b','as01_c', 'as02_a','as02_b','as02_c', 'rnn_a', 'rnn_b', 'rnn_c', 'cnn_a', 'cnn_b', 'cnn_c']:
            for ds_name in all_datasets:
                composite = "models/{}/{}_lag{}.pth".format(model_name, ds_name, lag)
                if composite not in all_models:
                    print("missing:", composite)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M4", action="store_const", const=True, default=False)
    parser.add_argument("--lag", action="store")
    args = parser.parse_args()
    main(m4=args.M4, lag=args.lag)