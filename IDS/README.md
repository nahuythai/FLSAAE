# Capture and extract traffic
`sudo cicflowmeter -i enp0s3 -c flows.csv`
# Listen and detect
`python3 ids.py --encoder_path=encoder_inseclab_10 --data_path=flows.csv --batch_size=100`
