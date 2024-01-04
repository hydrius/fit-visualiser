import streamlit as st
import pandas as pd
import numpy as np

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

import os

import warnings
warnings.filterwarnings('ignore')



env =   {"server": st.secrets["SERVER"],
            "token": st.secrets["TOKEN"],
            "chat_id": st.secrets["CHAT_ID"],
            "password": st.secrets["PASS"],
            "user": st.secrets["USER"],
            "database": st.secrets["DB"]
         }



#https://docs.streamlit.io/library/api-reference/charts/st.scatter_chart#


class DataVisualiser():
    def __init__(self, env):
        self.env = env


    def get_data(self):


        # Create a connection to the PostgreSQL database
        conn_str = f"postgresql+psycopg2://{self.env['user']}:{self.env['password']}@{self.env['server']}/{self.env['database']}"
        engine = create_engine(conn_str)

        query = f"SELECT * FROM fitness"
        data = pd.read_sql_query(query, engine)

        engine.dispose()
        

        # np.random.seed(42)  # Set seed for reproducibility
        # date_index = pd.date_range('2023-01-01', periods=365, freq='D')  # Daily frequency for one year
        # random_heart_minutes = np.random.randint(60, 150, size=len(date_index))

        # # Create DataFrame
        # data = {'com.google.heart_minutes': random_heart_minutes}
        # df = pd.DataFrame(data, index=date_index)
        # df['datetime'] = df.index  # Add a column with the datetime index

        return data

    def data_processing(self, data):
        #return data
        # Pivot the DataFrame
        pivot_df = data.pivot_table(index=data.datetime, columns='observation', values='value')
        
        # Resetting the index, if needed
        #pivot_df = pivot_df.reset_index(drop=True)

        return pivot_df



    def transition_matrix(self, data):

        low_intensity = 50
        rest = 10


        transition_matrix = [[0, 0, 0], #high low rest
                            [0, 0, 0], #low
                            [0, 0, 0]] #rest
                #print(data1)
        for i, hp in enumerate(data):
            if i > 1:
                if hp >= low_intensity:
                    current = "high"
                    if previous == "high":
                        transition_matrix[0][0]+=1
                    elif previous == "low":
                        transition_matrix[0][1]+=1
                    elif previous == "rest":
                        transition_matrix[0][2]+=1
                    
                if hp < low_intensity and hp >= rest: 
                    current = "low"
                    if previous == "high":
                        transition_matrix[1][0]+=1
                    elif previous == "low":
                        transition_matrix[1][1]+=1
                    elif previous == "rest":
                        transition_matrix[1][2]+=1
                    
                if hp < rest:
                    current = "rest"
                    if previous == "high":
                        transition_matrix[2][0]+=1
                    elif previous == "low":
                        transition_matrix[2][1]+=1
                    elif previous == "rest":
                        transition_matrix[2][2]+=1
                    
            else:
                current = "rest"
                previous = "rest"
            previous = current

        transition_matrix = pd.DataFrame(transition_matrix, columns=['High', 'Low', 'Rest'], index=['High', 'Low', 'Rest'])


        return transition_matrix


    def charting(self):

        raw = self.data_processing(self.get_data())
        data = raw.loc[pd.to_datetime('today') - pd.DateOffset(days=90):]
        date_from = pd.to_datetime('today') - pd.DateOffset(days=90)
        date_to = pd.to_datetime('today')- pd.DateOffset(days=1)
        #Layout
        st.set_page_config(
            page_title="Data Explorer",
            layout="centered",
            page_icon=":eyeglasses:") #wide centered
            #initial_sidebar_state="collapsed") #expande

        # SIDE BAR STUFF
        tabs_list = ["Weather", "Fitness"]

        low_intensity = 50
        rest = 10
        data['com.google.heart_minutes'] = data['com.google.heart_minutes'].fillna(0)

        matrix = self.transition_matrix(data['com.google.heart_minutes'].tolist())
        print(matrix)
        tabs = st.sidebar.selectbox('Tabs', tabs_list)
        
        st.title("Fitness Data")
        st.header(f"{date_from.strftime('%d-%m-%Y')} to {date_to.strftime('%d-%m-%Y')}", divider='rainbow') #TO DO

        data['High Intensity'] = np.where(data['com.google.heart_minutes'] >= low_intensity, data['com.google.heart_minutes'], np.nan)
        data['Low Intensity'] = np.where(((data['com.google.heart_minutes'] < low_intensity) & (data['com.google.heart_minutes'] >= rest)), data['com.google.heart_minutes'], np.nan)
        data['Rest'] = np.where(data['com.google.heart_minutes'] < rest, data['com.google.heart_minutes'], np.nan)

        summary = data[['High Intensity','Low Intensity', 'Rest']].describe().loc[['count', 'mean','min','max']]

        data1 = data.reset_index()

        data1['Week'] = data1['datetime'].dt.isocalendar().week
        mean_heart_points_per_week = data1.groupby('Week')['com.google.heart_minutes'].sum()
        mean_weekly_hp = mean_heart_points_per_week.mean()
        target_weekly_hp = mean_weekly_hp * 1.1
        
        st.subheader("Weekly Goals")
        st.write(f"Current Weekly Heart Points: {mean_heart_points_per_week.iloc[-1]: .0f}.")
        st.write(f"Mean Weekly Heart Points: {mean_weekly_hp: .0f}")
        st.write(f"Target Weekly Heart Points: {target_weekly_hp: .0f}")
        st.write(f"Target Daily Heart Points: {target_weekly_hp/7: .0f}")


        st.subheader("Daily Summaries")
        st.write(f"Daily Summary of Heart Points: High Intensity > {low_intensity}.")
        st.table(summary)

        st.write(f"Matrix of consecutive workouts by intensity.")
        st.write(matrix)
        st.divider()

        st.write(f"Heart Points over the previous 90 days.")
        st.scatter_chart(data[['High Intensity','Low Intensity', 'Rest']])


dv = DataVisualiser()
dv.charting()
