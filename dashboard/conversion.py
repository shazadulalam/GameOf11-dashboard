import datetime
# import SqlDatabase
from django.shortcuts import render
from django import forms as form
from . import mysqlDb as sql
from . import validation as val
from random import sample
from graphos.sources.simple import SimpleDataSource
from graphos.renderers.gchart import PieChart, LineChart, BarChart
import pandas as pd
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import numpy as np


def daterange(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + datetime.timedelta(n)

def conversionSegmentData(request):

    if request.POST.get('conversionSegmentsubmit') == 'conversionSegmentform':

        form_sdate = (request.POST.get('sdate'))
        form_edate = (request.POST.get('edate'))

        # print(form_sdate, "----------")

        datecheck = val.formDateValidation(request, form_sdate, form_edate, 'dashboard/users.html')
        if datecheck != 'valid':
            return datecheck
        
        sdate = form_sdate
        edate = (datetime.datetime.strptime(form_edate, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime(
            '%Y-%m-%d')
        sdate = datetime.datetime.strptime(sdate, '%Y-%m-%d').strftime('%b %d, %Y')
        edate = datetime.datetime.strptime(edate, '%Y-%m-%d').strftime('%b %d, %Y')
        # print(sdate, edate)
        #cash to coin conversion 
        sql_sDate = form_sdate
        sql_eDate = (datetime.datetime.strptime(form_edate, '%Y-%m-%d')).strftime(
        '%Y-%m-%d')
        
        page = request.GET.get('page')

        

        conversion = sql.getData("SELECT cast(created_at as date) as date, count(user_id)FROM `user_cash_log` WHERE request_type = 'conversion' \
                                    and status = 'success' and channel_type = 'cash to coin conversion' and created_at BETWEEN '"+ form_sdate +"' and '"+ form_edate +"' \
                                    group by cast(created_at as date)")

        conversion = [[date, user] for date, user in conversion]

        paginator = Paginator(conversion, 2)
        try:
            conversion = paginator.get_page(page)
        except PageNotAnInteger:
            conversion = paginator.get_page(1)
        except EmptyPage:
            conversion = paginator.get_page(paginator.num_pages)

        matchId = sql.getData("SELECT matches.id, matches.name, matches.match_time, count(contests.id) as contest_count FROM matches join contests ON matches.id = contests.match_id WHERE matches.is_published = 1 AND matches.match_time >= '"+ form_sdate +"' AND matches.match_time < '"+ form_edate +"' GROUP BY matches.id")

        # calculateData = fetchData(str(matchId))

        # print(matchId)

        mId = []
        df_processed = []
        counted_df = []
        for match_id, name, matchTime, contestCount in matchId:

            mid = profitMergin(str(match_id))
            mId.append(mid)
            df_processed.append([str(match_id), name, matchTime, contestCount])

        # print(df_processed)
            
        matchInfo = [(index[0], index[1], index[2], index[3]) for index in (df_processed)]
        print(matchInfo)
        
        columns = ['match_id', 'name', 'matchTime', 'totalcontestgiven']
        df = pd.DataFrame(df_processed, columns=columns)

        for data in mId:
            for frequency in data:
                counted_df.append([frequency[0], frequency[1], frequency[2], frequency[3], frequency[4]])

        totalMergin = [(data[0], data[1], data[2], data[3], data[4]) for data in (counted_df)]
        print(totalMergin)
        
        # totalMergin.insert(0,['Total Entry Amount', 'ProfitMergin'])
        
        # options = {"title": "Entry amount vs Profit Mergin"}
        # totalMergin_chart = LineChart(SimpleDataSource(data=totalMergin), options=options)

        coulumn1 = ['TotalEntryAmount', 'TotalSeat', 'TotalWinningAMount', 'TotalTeamCapacity', 'ProfitMergin']
        df1 = pd.DataFrame(counted_df, columns=coulumn1)

        final_df = pd.concat([df, df1], axis=1)
        
        datax = np.concatenate((matchInfo, totalMergin), axis=1)
        # final_df = final_df.to_html()
        print(datax)

        return render(request, 'dashboard/conversion.html', {'conversion': conversion, 'sdate': sdate, 'edate': edate, 'datax':datax })
    else:
        return render(request, 'dashboard/conversion.html')

def profitMergin(match_id):

    calculateData = sql.getData("SELECT SUM(Total_Per_Contest.Total_Entry_Amount_Per_Contest)/50 AS Total_Entry_Amount, SUM(Total_Per_Contest.Total_Seat_Per_Contest) AS Total_Seat, Sum(Total_Per_Contest.winning_amount) AS Total_Winning_Amount, \
        Sum(Total_Per_Contest.teams_capacity) AS Total_Team_Capacity, (SUM(Total_Per_Contest.Total_Entry_Amount_Per_Contest)/50 -Sum(Total_Per_Contest.winning_amount)) As Profit_Mergin \
            FROM (SELECT COUNT(U_T_C.contest_id)*C.entry_amount as Total_Entry_Amount_Per_Contest, COUNT(U_T_C.contest_id) As Total_Seat_Per_Contest, C.winning_amount, C.teams_capacity, C.id AS Contest_ID FROM user_team_contests as U_T_C \
                INNER JOIN (SELECT id, entry_amount, winning_amount,teams_capacity FROM contests WHERE match_id= "+ match_id +" and contest_type='paid') as C on U_T_C.contest_id = C.id GROUP BY U_T_C.contest_id) As Total_Per_Contest")


    return calculateData

