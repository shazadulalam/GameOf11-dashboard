import datetime
# import SqlDatabase
from django.shortcuts import render
from django import forms as form
from . import mysqlDb as sql
from . import validation as val
from random import sample
from graphos.sources.simple import SimpleDataSource
from graphos.renderers.gchart import PieChart, LineChart, BarChart
import numpy as np

def daterange(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + datetime.timedelta(n)

def userSegmentData(request):

    if request.POST.get('userSegmentsubmit') == 'userSegmentform':

        form_sdate = (request.POST.get('sdate'))
        form_edate = (request.POST.get('edate'))

        print(form_sdate, form_edate)

        datecheck = val.formDateValidation(request, form_sdate, form_edate, 'dashboard/users.html')
        if datecheck != 'valid':
            return datecheck
        
        sdate = form_sdate
        edate = (datetime.datetime.strptime(form_edate, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime(
            '%Y-%m-%d')
        sdate = datetime.datetime.strptime(sdate, '%Y-%m-%d').strftime('%b %d, %Y')
        edate = datetime.datetime.strptime(edate, '%Y-%m-%d').strftime('%b %d, %Y')

        print(sdate, edate)
        #per day item purchase
        data = sql.getData("SELECT cast(created_at as date) as date,count(id) FROM `users` WHERE created_at between '"+ form_sdate +"' \
                            and '"+ form_edate +"' group by cast(created_at as date) ")
        # print(data)
        datasort_sDate = datetime.datetime.strptime(form_sdate, '%Y-%m-%d')
        datasort_eDate = datetime.datetime.strptime(form_edate, '%Y-%m-%d')

        all_date = [dt.strftime('%Y-%m-%d') for dt in daterange(datasort_sDate, datasort_eDate)]
        sortedDate = sorted(all_date)
        data = {date.strftime('%Y-%m-%d') if type(date) != str else date: (user)
                              for date, user in data}
                
        # print(data)
        for date in sortedDate:
            if date not in data:
                data[date] = (0, 0, 0)
        
        # print(sortedDate)
        userData = [[date, data[date]] for date in sortedDate]
        # print(userData)
        userData.insert(0, ['Date', 'TotalUser'])
        # print(itemid_final)
        userChart = BarChart(SimpleDataSource(data=userData))
        # print(userChart)
        totalUser = sql.getData("select count(id) from `users`")
        user = 0
        for i in totalUser:
            user = i[0]
        userCash = sql.getData("SELECT cast(created_at as date) as date,sum(total_cash) FROM `users_metadata` WHERE created_at between '"+ form_sdate +"' \
                            and '"+ form_edate +"' group by cast(created_at as date) ")
        print(userCash)
        userCashDetails = [[date, amount] for date, amount in userCash]
        userCashDetails.insert(0, ['Date', 'Amount'])
        userCash_chart = PieChart(SimpleDataSource(data=userCashDetails))

        #daily transaction by users
        dailyIn = sql.getData("SELECT CAST(created_at AS DATE) AS DATE, SUM(recharge_amount) FROM `user_coins_log` WHERE source_type = 'ghoori-bkash' \
                                AND created_at BETWEEN '"+ form_sdate +"' AND '"+ form_edate +"' GROUP BY CAST(created_at AS DATE)")
        
        dailyIn = [[date, amount] for date, amount in dailyIn]
        print(dailyIn)
        dailyIn.insert(0, ['Date', 'Amount'])
        dailyInChart = LineChart(SimpleDataSource(data=dailyIn))

        uniqueUser = sql.getData("SELECT count(distinct(user_id)) as cnt FROM user_coins_log where recharge_amount>0")
        for i in uniqueUser:
            uniqueUser = i[0]
        contestUniqueUser = sql.getData("SELECT count(distinct(user_id)) FROM user_coins_log where contest_id>0")
        for i in contestUniqueUser:
            contestUniqueUser = i[0]
        uniqueUserPrize = sql.getData("SELECT count(distinct(user_id)) FROM user_cash_log where request_type='withdraw' and status!='canceled'")
        for i in uniqueUserPrize:
            uniqueUserPrize = i[0]
        faileduser = sql.getData("SELECT 1,id FROM users WHERE id IN(SELECT user_coins_log.user_id FROM user_coins_log \
                    WHERE date(user_coins_log.created_at)<= '2020-12-31' AND user_coins_log.user_id IN (SELECT users.id \
                    FROM users WHERE date(users.created_at)>='2019-02-14' AND date(users.created_at)<='2020-12-31') and \
                    user_coins_log.transaction_type='debited' GROUP BY user_coins_log.user_id HAVING (SUM(user_coins_log.coin)< 1000  AND SUM(user_coins_log.coin)>0)))")


        return render(request, 'dashboard/users.html',{'userChart': userChart, 'cashChart': userCash_chart, 'd_chart': dailyInChart, 'playedContestUnique': contestUniqueUser, 
                                                        'totalUser': user,'sdate': sdate, 'edate': edate, 'uniqueUser': uniqueUser, 'uniqueUserPrize': uniqueUserPrize})
    else:
        return render(request, 'dashboard/users.html')
    
def cohorts(request):
    if request.POST.get('cohortsubmit') == 'cohortsvalue':
        month = request.POST.get('month')
        year = request.POST.get('year')
        return cohortsAction(request, month, year)


    else:
        return cohortsAction(request, '1', '2019')


def cohortsAction(request, month, year):
    mdateS, myearS = int(month), year


    print(mdateS, myearS)
    sdate1 = myearS + '/' + str(mdateS) + '/1'
    edate1 = myearS + '/' + str(mdateS + 1) + '/1'
    print(sdate1)


    sdate2 = myearS + '/' + str(mdateS + 1) + '/1'
    edate2 = myearS + '/' + str(mdateS + 2) + '/1'



    sdate3 = myearS + '/' + str(mdateS + 2) + '/1'
    edate3 = myearS + '/' + str(mdateS + 3) + '/1'



    sdate4 = myearS + '/' + str(mdateS + 3) + '/1'
    edate4 = myearS + '/' + str(mdateS + 4) + '/1'


    dataqm0 = sql.getData("SELECT DISTINCT(id) FROM users WHERE date(created_at) BETWEEN '"+ sdate1 +"' and '"+ edate1 +"'")
    dataqm = sql.getData("select DISTINCT(users.id) from users inner join user_team_contests on users.id=user_team_contests.user_id \
    where date(user_team_contests.created_at) between '"+ sdate2 +"' and '"+ edate2 +"' and date(users.created_at) between '"+ sdate1 +"' and '"+ edate1 +"'")
    dataqm1 = sql.getData("select DISTINCT(users.id) from users inner join user_team_contests_archive on users.id=user_team_contests_archive.user_id \
    where date(user_team_contests_archive.created_at) between '"+ sdate2 +"' and '"+ edate2 +"' and date(users.created_at) between '"+ sdate1 +"' and '"+ edate1 +"'")

    dataqm2 = sql.getData("SELECT DISTINCT(id) FROM users WHERE date(created_at) BETWEEN '"+ sdate2 +"' and '"+ edate2 +"'")

    dataqm2a = sql.getData("select DISTINCT(users.id) from users inner join user_team_contests on users.id=user_team_contests.user_id \
    where date(user_team_contests.created_at) between '"+ sdate3 +"' and '"+ edate3 +"' and date(users.created_at) between '"+ sdate2 +"' and '"+ edate2 +"'")
    dataqm2b = sql.getData("select DISTINCT(users.id) from users inner join user_team_contests_archive on users.id=user_team_contests_archive.user_id \
    where date(user_team_contests_archive.created_at) between '"+ sdate3 +"' and '"+ edate3 +"' and date(users.created_at) between '"+ sdate2 +"' and '"+ edate2 +"'")


    dataqm3 = sql.getData("SELECT DISTINCT(id) FROM users WHERE date(created_at) BETWEEN '"+ sdate3 +"' and '"+ edate3 +"'")
    dataqm3a = sql.getData("select DISTINCT(users.id) from users inner join user_team_contests on users.id=user_team_contests.user_id \
    where date(user_team_contests.created_at) between '"+ sdate4 +"' and '"+ edate4 +"' and date(users.created_at) between '"+ sdate3 +"' and '"+ edate3 +"'")
    dataqm3b = sql.getData("select DISTINCT(users.id) from users inner join user_team_contests_archive on users.id=user_team_contests_archive.user_id \
    where date(user_team_contests_archive.created_at) between '"+ sdate4 +"' and '"+ edate4 +"' and date(users.created_at) between '"+ sdate3 +"' and '"+ edate3 +"'")



    dataqm4 = sql.getData("SELECT DISTINCT(id) FROM users WHERE date(created_at) BETWEEN '"+ sdate4 +"' and '"+ edate4 +"'")

    # habijabi = sql.getData("select id from users where date(created_at) BETWEEN '2020-11-01' and '2020-11-30'")
    # habijabi = [customerid[0] for customerid in habijabi]
    # print(habijabi)
    m_cus = [customerid[0] for customerid in dataqm0]
    m_customers = [customerid[0] for customerid in dataqm]
    m0_customers = [customerid[0] for customerid in dataqm1]
    m0_cus = np.concatenate([m_customers, m0_customers])

    m1_customers = [customerid[0] for customerid in dataqm2]
    m1_customersA = [customerid[0] for customerid in dataqm2a]
    m1_customersB = [customerid[0] for customerid in dataqm2b]
    m1_final = np.concatenate([m1_customersA, m1_customersB])

    m2_customers = [customerid[0] for customerid in dataqm3]
    m2_customersA = [customerid[0] for customerid in dataqm3a]
    m2_customersB = [customerid[0] for customerid in dataqm3b]
    m2_final = np.concatenate([m2_customersA, m2_customersB])


    m3_customers = [customerid[0] for customerid in dataqm4]


    m0_m1 = set(m_cus).intersection(m0_cus)
    m0_m2 = set(m_cus).intersection(m1_final)
    m0_m3 = set(m_cus).intersection(m2_final)
    # print("___________", len(m_cus),len(m0_cus), len(m_customers), len(m0_customers))


    datam1 = [len(m_cus), len(m0_m1), len(m0_m2), len(m0_m3)]
    # print(datam1)
    m1_m2 = set(m1_customers).intersection(m1_final)
    m1_m3 = set(m1_customers).intersection(m2_final)

    datam2 = [len(m1_customers), len(m1_m2), len(m1_m3)]

    m2_m3 = set(m2_customers).intersection(m2_final)
    datam3 = [len(m2_customers), len(m2_m3)]

    datam4 = len(m3_customers)
    # print("___________",datam2)
    return render(request,
                  'dashboard/cohorts.html',
                  {'isact_cohorts': 'active',
                   "htmltitle": "Cohort Analysis", 'sdate': sdate1,

                   'edate': sdate4, 'datam1': datam1, 'datam2': datam2,
                   'datam3': datam3, 'datam4': datam4

                   })

    
