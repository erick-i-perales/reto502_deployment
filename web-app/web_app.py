import requests
import pandas as pd
import streamlit as st

output_pipeline_path = 'preprocessed_data.csv'

def post_request(url, files, params):
    '''
    Args:
        None

    Returns:
        data [dictionary] : Response from the API (JSON like response)

    '''
    try:
        # Make a GET request to the API endpoint using requests.get()
        response = requests.post(url, files=files, params=params)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print('Error:', response.status_code)
            return None
    except requests.exceptions.RequestException as e:
  
        # Handle any network-related errors or exceptions
        print('Error:', e)
        return None

def main():

    # Web App
    # --------------------------------------------------------------------------   

    st.title('Income Forecast')
    st.write('Web app demo interacting with API for time series prediction.')

    # Input Parameters
    # --------------------------------------------------------------------------  
    preprocessed_data = pd.read_csv(output_pipeline_path)
    preprocessed_data['date'] = pd.to_datetime(preprocessed_data['date'])

    # st.dataframe(preprocessed_data, hide_index=True, use_container_width=True)

    company = st.selectbox('Company',
                            preprocessed_data.columns[1:])

    agg_options = {'By day':'daily', 'By week':'weekly', 'By month':'monthly'}

    aggregation = st.selectbox('Compute predicions:',
                            agg_options.keys())
    aggregation = agg_options[aggregation]

    num_of_predictions = st.number_input("Number of predictions to compute:", step=1, format="%d")

    series = preprocessed_data[['date', company]]

    # Creatin JSON file from chosen company series
    # --------------------------------------------------------------------------  

    filtered_df = series.dropna(subset=[company])
    filtered_df.rename(columns={company: 'data'}, inplace=True)
    json_data = filtered_df.to_json(orient='records')#, date_format='iso', date_unit='ms')


    # API Endpoint
    # --------------------------------------------------------------------------   

    url = f'http://127.0.0.1:10000/uploadfile/predict_{aggregation}'
    params = {"num_of_predictions": num_of_predictions}


    # API Call
    # --------------------------------------------------------------------------   

    if st.button('Compute Predictions'):

        with open('data.json', 'w') as json_file:
            json_file.write(json_data)
        files = {'file': open('data.json' ,'rb')}
        results = post_request(url, files, params)

        if results:
            forecast = results['forecast']
            metrics = results['metrics']

            forecast_df = pd.DataFrame(list(forecast.items()), 
                                    columns=['Date', 'Forecast'])
            metrics = [{'Metric': key, ' ': value} for key, value in metrics.items()]
            
            metrics_df = pd.DataFrame(metrics)
            
            st.dataframe(forecast_df, hide_index=True, use_container_width=True)

            st.write('Metrics')
            st.dataframe(metrics, hide_index=True, use_container_width=True)

            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Forecast as CSV",
                data=csv,
                file_name=f'forecast_{company}_{aggregation}.csv',
                mime='text/csv'
            )
            
        else:
            print('Failed to fetch data from API.')

if __name__ == '__main__':
    main()